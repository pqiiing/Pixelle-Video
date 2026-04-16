# Copyright (C) 2025 AIDC-AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Script Extractor Service

Downloads video from platform URLs (Bilibili, YouTube, Douyin, etc.),
then sends to LLM multimodal model for script extraction.

- Douyin: Playwright subprocess (headless browser) to get video URL + cookies
- Other platforms: yt-dlp
"""

import base64
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import httpx
from loguru import logger
from openai import AsyncOpenAI

from pixelle_video.config import config_manager


EXTRACT_PROMPT = """请仔细观看这个视频，提取视频中所有的口播文案/旁白/解说词。

要求：
1. 完整提取视频中说话人的所有内容，不要遗漏
2. 保持原始语言，不要翻译
3. 按说话顺序排列，用换行分段
4. 只输出文案内容本身，不要加任何标注、时间戳或额外说明"""

MAX_VIDEO_SIZE_MB = 10
MAX_DURATION_SEC = 600

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)

_DOUYIN_HOSTS = ("v.douyin.com", "www.douyin.com", "douyin.com", "www.iesdouyin.com")

# Playwright helper script — executed in a separate process to avoid
# asyncio event-loop conflicts on Windows.
_PW_HELPER = r'''
import json, sys
from playwright.sync_api import sync_playwright

url = sys.argv[1]
with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    ctx = browser.new_context(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36")
    page = ctx.new_page()

    try:
        page.goto(url, wait_until="domcontentloaded", timeout=30000)
        page.wait_for_timeout(8000)
    except Exception as e:
        print(json.dumps({"error": str(e), "video_url": "", "title": ""}))
        browser.close()
        sys.exit(0)

    video_url = page.evaluate("""() => {
        const s = document.querySelector('video source');
        if (s) return s.src || s.getAttribute('src') || '';
        const v = document.querySelector('video');
        if (v) return v.src || v.currentSrc || '';
        return '';
    }""")

    title = page.evaluate("() => document.title || ''")

    browser.close()
    print(json.dumps({"video_url": video_url, "title": title, "error": ""}))
'''


class ScriptExtractorService:
    """
    Extract script from video platform URLs using LLM.

    Pipeline: platform URL → download video → base64 encode → LLM multimodal → text
    """

    def __init__(self):
        self._cache_dir = Path(tempfile.gettempdir()) / "pixelle_video_cache"
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._pw_helper_path = self._cache_dir / "_pw_helper.py"
        self._pw_helper_path.write_text(_PW_HELPER, encoding="utf-8")

    def _create_client(self) -> tuple[AsyncOpenAI, str]:
        llm_config = config_manager.get_llm_config()
        api_key = llm_config["api_key"] or "dummy-key"
        base_url = llm_config["base_url"] or None
        model = llm_config["model"] or "gpt-4o"

        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url

        return AsyncOpenAI(**client_kwargs), model

    @staticmethod
    def parse_url(raw_input: str) -> str:
        """Extract a valid URL from raw input (e.g. Douyin share text)."""
        raw_input = raw_input.strip()
        match = re.search(r'https?://\S+', raw_input)
        if match:
            return match.group(0).rstrip('/')
        raise ValueError("未找到有效链接，请粘贴包含 http(s):// 的视频链接")

    @staticmethod
    def _is_douyin(url: str) -> bool:
        from urllib.parse import urlparse
        return urlparse(url).hostname in _DOUYIN_HOSTS

    def _url_hash(self, url: str) -> str:
        return hashlib.md5(url.encode()).hexdigest()[:12]

    # ── Douyin: Playwright-based ────────────────────────────────────────

    def _run_playwright_helper(self, url: str) -> dict:
        """Run Playwright in a subprocess to visit a page and extract video URL.

        This avoids asyncio event-loop conflicts on Windows (Python 3.14+).
        """
        logger.info("Launching Playwright subprocess...")
        result = subprocess.run(
            [sys.executable, str(self._pw_helper_path), url],
            capture_output=True, text=True, timeout=90,
        )

        if result.returncode != 0:
            raise RuntimeError(f"Playwright 子进程失败：{result.stderr[-300:]}")

        stdout = result.stdout.strip()
        if not stdout:
            raise RuntimeError("Playwright 子进程无输出")

        # Take the last JSON line (Playwright may print warnings before)
        last_line = stdout.split("\n")[-1]
        data = json.loads(last_line)

        if data.get("error"):
            raise RuntimeError(f"Playwright 页面加载失败：{data['error']}")

        return data

    def _download_douyin(self, url: str) -> Path:
        """Download Douyin video using Playwright to extract direct video URL."""
        url_hash = self._url_hash(url)
        output_path = self._cache_dir / f"video_{url_hash}.mp4"

        if output_path.exists():
            logger.info(f"Using cached video: {output_path}")
            return output_path

        logger.info(f"Downloading Douyin video: {url}")
        data = self._run_playwright_helper(url)
        video_url = data.get("video_url", "")

        if not video_url:
            raise RuntimeError(
                "无法从抖音页面获取视频地址。"
                "请确认链接有效，且 Playwright Chromium 已安装（运行 playwright install chromium）"
            )

        logger.info(f"Douyin video URL resolved, downloading...")
        headers = {"User-Agent": _USER_AGENT, "Referer": "https://www.douyin.com/"}

        with httpx.Client(headers=headers, timeout=60, follow_redirects=True) as client:
            resp = client.get(video_url)
            resp.raise_for_status()
            output_path.write_bytes(resp.content)

        size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"Douyin video downloaded: {size_mb:.1f} MB")

        if size_mb > MAX_VIDEO_SIZE_MB:
            output_path = self._compress_video(output_path)

        return output_path

    def _get_douyin_info(self, url: str) -> dict:
        """Get Douyin video info via Playwright subprocess."""
        try:
            data = self._run_playwright_helper(url)
            title = data.get("title", "")
            # Clean up Douyin title (format: "描述 - 抖音")
            if " - " in title:
                title = title.rsplit(" - ", 1)[0]
            return {"title": title, "duration": 0, "uploader": ""}
        except Exception as e:
            logger.warning(f"Douyin info failed: {e}")
            return {"title": "", "duration": 0, "uploader": ""}

    # ── Generic: yt-dlp ─────────────────────────────────────────────────

    def _download_ytdlp(self, url: str) -> Path:
        """Download video using yt-dlp (YouTube, Bilibili, etc.)."""
        import yt_dlp

        url_hash = self._url_hash(url)
        output_path = self._cache_dir / f"video_{url_hash}.mp4"

        if output_path.exists():
            logger.info(f"Using cached video: {output_path}")
            return output_path

        logger.info(f"Downloading video via yt-dlp: {url}")

        ydl_opts = {
            "format": "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best",
            "outtmpl": str(output_path),
            "merge_output_format": "mp4",
            "quiet": True,
            "no_warnings": True,
            "socket_timeout": 30,
            "retries": 3,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
        except Exception as e:
            if output_path.exists():
                output_path.unlink()
            raise RuntimeError(f"视频下载失败：{e}") from e

        if not output_path.exists():
            raise RuntimeError("下载完成但文件不存在")

        size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"Downloaded: {size_mb:.1f} MB")

        if size_mb > MAX_VIDEO_SIZE_MB:
            output_path = self._compress_video(output_path)

        return output_path

    # ── Public interface ────────────────────────────────────────────────

    def download_video(self, url: str) -> Path:
        """Download video, auto-selecting the best method for the platform."""
        url = self.parse_url(url)
        if self._is_douyin(url):
            return self._download_douyin(url)
        return self._download_ytdlp(url)

    def get_video_info(self, url: str) -> dict:
        """Get video metadata without downloading."""
        url = self.parse_url(url)

        if self._is_douyin(url):
            return self._get_douyin_info(url)

        import yt_dlp
        ydl_opts = {"quiet": True, "no_warnings": True, "skip_download": True}
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
            return {
                "title": info.get("title", ""),
                "duration": info.get("duration", 0),
                "uploader": info.get("uploader", ""),
            }
        except Exception:
            return {"title": "", "duration": 0, "uploader": ""}

    def _compress_video(self, video_path: Path) -> Path:
        """Compress video to fit LLM API size limits."""
        compressed = video_path.with_name(video_path.stem + "_small.mp4")
        if compressed.exists():
            return compressed

        logger.info("Compressing video for LLM input...")
        cmd = [
            "ffmpeg", "-i", str(video_path),
            "-vf", "scale=-2:480",
            "-c:v", "libx264", "-crf", "28",
            "-c:a", "aac", "-b:a", "64k",
            "-t", str(MAX_DURATION_SEC),
            "-y", str(compressed),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            logger.warning(f"Compression failed, using original: {result.stderr[:200]}")
            return video_path

        size_mb = compressed.stat().st_size / (1024 * 1024)
        logger.info(f"Compressed: {size_mb:.1f} MB")
        return compressed

    def _video_to_base64(self, video_path: Path) -> str:
        with open(video_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    async def _extract_via_file_upload(
        self, video_path: Path, client: AsyncOpenAI, model: str
    ) -> str:
        """Upload video file via Files API, then reference file_id in chat."""
        logger.info(f"Uploading video via Files API: {video_path}")

        file_obj = await client.files.create(
            file=open(video_path, "rb"),
            purpose="file-extract",
        )
        logger.info(f"File uploaded: id={file_obj.id}")

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": {"file_id": file_obj.id},
                    },
                    {"type": "text", "text": EXTRACT_PROMPT},
                ],
            }
        ]

        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=4096,
            temperature=0.3,
        )
        return (response.choices[0].message.content or "").strip()

    async def _extract_via_base64(
        self, video_path: Path, client: AsyncOpenAI, model: str
    ) -> str:
        """Send video as base64 data-uri (original approach)."""
        video_b64 = self._video_to_base64(video_path)
        data_uri = f"data:video/mp4;base64,{video_b64}"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video_url", "video_url": {"url": data_uri}},
                    {"type": "text", "text": EXTRACT_PROMPT},
                ],
            }
        ]

        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=4096,
            temperature=0.3,
        )
        return (response.choices[0].message.content or "").strip()

    async def extract_script(self, url: str) -> str:
        """
        Full pipeline: parse URL → download → compress → LLM multimodal → text

        Strategy:
          1. Try file upload (Files API) — no size limit on data-uri
          2. Fall back to base64 data-uri if upload not supported
        """
        url = self.parse_url(url)
        logger.info(f"Extracting script from: {url}")

        video_path = self.download_video(url)

        size_mb = video_path.stat().st_size / (1024 * 1024)
        if size_mb > MAX_VIDEO_SIZE_MB:
            video_path = self._compress_video(video_path)

        client, model = self._create_client()

        # Prefer base64 data-uri (widely supported by DashScope / OpenAI / Gemini).
        # File-upload via Files API only works for document extraction on DashScope,
        # NOT for video analysis, so it is used only as a last resort.
        try:
            script = await self._extract_via_base64(video_path, client, model)
            logger.info(f"Script extracted (base64): {len(script)} chars")
            return script
        except Exception as b64_err:
            logger.warning(f"Base64 approach failed, trying file upload: {b64_err}")

        try:
            script = await self._extract_via_file_upload(video_path, client, model)
            logger.info(f"Script extracted (file upload): {len(script)} chars")
            return script
        except Exception as e:
            logger.error(f"Script extraction failed: {e}")
            raise RuntimeError(
                f"文案提取失败：{e}\n\n"
                "请确认 LLM 配置使用的是支持视频输入的多模态模型（如 Qwen-VL、Gemini、GPT-4o）"
            ) from e
