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
Super Agent Pipeline UI

Full workflow: Reference Study → Rewrite Script → Voice Generation →
Video Generation → One-click Assembly.
"""

import json
import os
import time
from pathlib import Path
from typing import Any

import httpx
import streamlit as st
from loguru import logger

from web.i18n import tr, get_language
from web.pipelines.base import PipelineUI, register_pipeline_ui
from web.components.content_input import render_bgm_section, render_version_info
from web.components.script_extract import render_script_extract
from web.utils.async_helpers import run_async
from web.utils.streamlit_helpers import check_and_warn_selfhost_workflow
from pixelle_video.config import config_manager
from pixelle_video.utils.os_util import create_task_output_dir


# ── LLM prompt templates for script rewriting ────────────────────────

_REWRITE_TEMPLATES = {
    "default": (
        "请将以下文案进行改写润色，保持原意但让表达更流畅自然、更有感染力，"
        "适合短视频口播。字数控制在{word_limit}字左右。\n\n原文案：\n{script}"
    ),
    "hook": (
        "请将以下文案改写为爆款短视频风格，开头必须用一个强有力的钩子吸引观众停留，"
        "语言口语化、节奏感强。字数控制在{word_limit}字左右。\n\n原文案：\n{script}"
    ),
    "story": (
        "请将以下文案改写为故事型风格，用讲故事的方式娓娓道来，"
        "引起观众共鸣。字数控制在{word_limit}字左右。\n\n原文案：\n{script}"
    ),
    "list": (
        "请将以下文案改写为清单型风格，条理清晰、重点突出，"
        "用数字列表呈现核心要点。字数控制在{word_limit}字左右。\n\n原文案：\n{script}"
    ),
}

_LEGAL_PROMPT = (
    "你是一名专业的内容合规审核员。请审核以下短视频文案是否存在以下法律风险：\n"
    "1. 虚假宣传或夸大功效\n2. 侵犯知识产权\n3. 涉及敏感政治内容\n"
    "4. 违反广告法（如使用绝对化用语）\n5. 其他潜在法律风险\n\n"
    "请给出审核结论和修改建议。\n\n文案内容：\n{script}"
)

_IP_PROMPT = (
    "你是一名专业的 IP 内容创作者。根据以下 IP 人设描述，"
    "创作一段适合短视频口播的文案，风格必须贴合人设特点。\n\n"
    "IP 人设：\n{persona}"
)

_MARKETING_PROMPT = (
    "你是一名顶级短视频营销文案策划师。根据以下产品/品牌信息，"
    "创作一段有吸引力、有转化力的短视频营销文案。\n\n"
    "产品信息：\n{product_info}"
)


class SuperAgentPipelineUI(PipelineUI):
    """
    Super Agent Pipeline UI.

    Full 5-step workflow matching the reference screenshot layout.
    """
    name = "super_agent"
    icon = "🧠"

    @property
    def display_name(self):
        return tr("pipeline.super_agent.name")

    @property
    def description(self):
        return tr("pipeline.super_agent.description")

    def render(self, pixelle_video: Any):
        left_col, middle_col, right_col = st.columns([1, 1, 1])

        with left_col:
            self._render_step1_reference(pixelle_video)
            self._render_step2_rewrite(pixelle_video)

        with middle_col:
            self._render_step3_voice(pixelle_video)
            self._render_step4_video(pixelle_video)

        with right_col:
            self._render_step5_assembly(pixelle_video)

    # ══════════════════════════════════════════════════════════════════
    # Step 1: 学习对标
    # ══════════════════════════════════════════════════════════════════
    def _render_step1_reference(self, pixelle_video: Any):
        with st.container(border=True):
            st.markdown(f"**{tr('super_agent.step1.title')}**")

            tab_extract, tab_ip, tab_marketing = st.tabs([
                tr("super_agent.step1.tab_extract"),
                tr("super_agent.step1.tab_ip"),
                tr("super_agent.step1.tab_marketing"),
            ])

            with tab_extract:
                script_params = render_script_extract(key_prefix="sa_")
                if script_params.get("extracted_script"):
                    st.session_state["sa_original_script"] = script_params["extracted_script"]

            with tab_ip:
                ip_text = st.text_area(
                    "IP",
                    placeholder=tr("super_agent.step1.ip_placeholder"),
                    height=120,
                    help=tr("super_agent.step1.ip_help"),
                    key="sa_ip_input",
                    label_visibility="collapsed",
                )
                if st.button(
                    tr("super_agent.step1.generate_btn"),
                    key="sa_ip_generate",
                    width="stretch",
                    disabled=not ip_text.strip(),
                ):
                    self._call_llm(
                        pixelle_video,
                        _IP_PROMPT.format(persona=ip_text.strip()),
                        target_key="sa_original_script",
                    )

            with tab_marketing:
                marketing_text = st.text_area(
                    "Marketing",
                    placeholder=tr("super_agent.step1.marketing_placeholder"),
                    height=120,
                    help=tr("super_agent.step1.marketing_help"),
                    key="sa_marketing_input",
                    label_visibility="collapsed",
                )
                if st.button(
                    tr("super_agent.step1.generate_btn"),
                    key="sa_marketing_generate",
                    width="stretch",
                    disabled=not marketing_text.strip(),
                ):
                    self._call_llm(
                        pixelle_video,
                        _MARKETING_PROMPT.format(product_info=marketing_text.strip()),
                        target_key="sa_original_script",
                    )

            st.markdown(f"**{tr('super_agent.step1.original_script')}**")
            st.text_area(
                "script_display",
                placeholder=tr("script_extract.url_placeholder"),
                height=150,
                key="sa_original_script",
                label_visibility="collapsed",
            )

    # ══════════════════════════════════════════════════════════════════
    # Step 2: 改写文案
    # ══════════════════════════════════════════════════════════════════
    def _render_step2_rewrite(self, pixelle_video: Any):
        with st.container(border=True):
            st.markdown(f"**{tr('super_agent.step2.title')}**")

            template_options = {
                "default": tr("super_agent.step2.template_default"),
                "hook": tr("super_agent.step2.template_hook"),
                "story": tr("super_agent.step2.template_story"),
                "list": tr("super_agent.step2.template_list"),
            }

            t_col, w_col = st.columns(2)
            with t_col:
                template_key = st.selectbox(
                    tr("super_agent.step2.template_label"),
                    options=list(template_options.keys()),
                    format_func=lambda x: template_options[x],
                    key="sa_rewrite_template",
                )
            with w_col:
                word_limit = st.selectbox(
                    tr("super_agent.step2.word_limit_label"),
                    options=[100, 200, 300, 500, 800],
                    index=2,
                    format_func=lambda x: f"{x}字",
                    key="sa_word_limit",
                )

            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                rewrite_clicked = st.button(
                    tr("super_agent.step2.rewrite_btn"),
                    width="stretch",
                    type="primary",
                    key="sa_rewrite_btn",
                )
            with btn_col2:
                legal_clicked = st.button(
                    tr("super_agent.step2.legal_btn"),
                    width="stretch",
                    key="sa_legal_btn",
                )

            if rewrite_clicked:
                original = st.session_state.get("sa_original_script", "")
                if not original.strip():
                    st.warning(tr("super_agent.step2.no_script"))
                else:
                    prompt = _REWRITE_TEMPLATES[template_key].format(
                        word_limit=word_limit, script=original
                    )
                    self._call_llm(pixelle_video, prompt, target_key="sa_rewritten_script")

            if legal_clicked:
                text_to_check = st.session_state.get(
                    "sa_rewritten_script",
                    st.session_state.get("sa_original_script", ""),
                )
                if not text_to_check.strip():
                    st.warning(tr("super_agent.step2.no_script"))
                else:
                    with st.spinner(tr("super_agent.step2.legal_checking")):
                        try:
                            result = run_async(
                                pixelle_video.llm(
                                    _LEGAL_PROMPT.format(script=text_to_check)
                                )
                            )
                            st.session_state["sa_legal_result"] = result
                        except Exception as e:
                            st.error(tr("super_agent.step2.legal_error", error=str(e)))

            st.markdown(f"**{tr('super_agent.step2.rewrite_result')}**")
            st.text_area(
                "rewrite_display",
                placeholder=tr("super_agent.step2.rewrite_placeholder"),
                height=150,
                key="sa_rewritten_script",
                label_visibility="collapsed",
            )

            legal_result = st.session_state.get("sa_legal_result")
            if legal_result:
                with st.expander(tr("super_agent.step2.legal_result"), expanded=False):
                    st.markdown(legal_result)

    # ══════════════════════════════════════════════════════════════════
    # Step 3: 声音生成
    # ══════════════════════════════════════════════════════════════════
    def _render_step3_voice(self, pixelle_video: Any):
        with st.container(border=True):
            st.markdown(f"**{tr('super_agent.step3.title')}**")

            from pixelle_video.tts_voices import EDGE_TTS_VOICES, get_voice_display_name

            comfyui_config = config_manager.get_comfyui_config()
            tts_config = comfyui_config["tts"]
            local_config = tts_config.get("local", {})
            saved_voice = local_config.get("voice", "zh-CN-YunjianNeural")
            saved_speed = local_config.get("speed", 1.2)

            voice_options = []
            voice_ids = []
            default_voice_index = 0
            for idx, vc in enumerate(EDGE_TTS_VOICES):
                vid = vc["id"]
                voice_options.append(get_voice_display_name(vid, tr, get_language()))
                voice_ids.append(vid)
                if vid == saved_voice:
                    default_voice_index = idx

            selected_display = st.selectbox(
                tr("super_agent.step3.voice_label"),
                voice_options,
                index=default_voice_index,
                key="sa_voice_select",
            )
            selected_voice = voice_ids[voice_options.index(selected_display)]

            speed_col, emotion_col = st.columns(2)
            with speed_col:
                tts_speed = st.slider(
                    tr("super_agent.step3.speed_label"),
                    min_value=0.5, max_value=2.0,
                    value=saved_speed, step=0.1,
                    format="%.1f",
                    key="sa_tts_speed",
                )
            with emotion_col:
                emotion_options = {
                    "normal": tr("super_agent.step3.emotion_normal"),
                    "happy": tr("super_agent.step3.emotion_happy"),
                    "sad": tr("super_agent.step3.emotion_sad"),
                    "angry": tr("super_agent.step3.emotion_angry"),
                    "gentle": tr("super_agent.step3.emotion_gentle"),
                }
                st.selectbox(
                    tr("super_agent.step3.emotion_label"),
                    options=list(emotion_options.keys()),
                    format_func=lambda x: emotion_options[x],
                    key="sa_emotion",
                )

            ref_audio_file = st.file_uploader(
                tr("super_agent.step3.upload_ref"),
                type=["mp3", "wav", "flac", "m4a", "aac", "ogg"],
                help=tr("super_agent.step3.upload_ref_help"),
                key="sa_ref_audio",
            )
            ref_audio_path = None
            if ref_audio_file is not None:
                st.audio(ref_audio_file)
                temp_dir = Path("temp")
                temp_dir.mkdir(exist_ok=True)
                ref_audio_path = temp_dir / f"sa_ref_{ref_audio_file.name}"
                with open(ref_audio_path, "wb") as f:
                    f.write(ref_audio_file.getbuffer())

            if st.button(
                tr("super_agent.step3.generate_btn"),
                type="primary",
                width="stretch",
                key="sa_tts_generate",
            ):
                script_text = st.session_state.get(
                    "sa_rewritten_script",
                    st.session_state.get("sa_original_script", ""),
                )
                if not script_text.strip():
                    st.warning(tr("super_agent.step3.no_text"))
                else:
                    with st.spinner(tr("super_agent.step3.generating")):
                        try:
                            task_dir, _ = create_task_output_dir()
                            audio_output = os.path.join(task_dir, "narration.mp3")

                            tts_kwargs = {
                                "text": script_text,
                                "output_path": audio_output,
                                "inference_mode": "local",
                                "voice": selected_voice,
                                "speed": tts_speed,
                            }
                            if ref_audio_path:
                                tts_kwargs["inference_mode"] = "comfyui"
                                tts_kwargs["workflow"] = "runninghub/tts_index2.json"
                                tts_kwargs["ref_audio"] = str(ref_audio_path)
                                tts_kwargs.pop("voice", None)
                                tts_kwargs.pop("speed", None)

                            result_path = run_async(pixelle_video.tts(**tts_kwargs))
                            st.session_state["sa_audio_path"] = result_path or audio_output
                            st.session_state["sa_task_dir"] = task_dir
                            st.success(tr("super_agent.step3.success"))
                            st.rerun()
                        except Exception as e:
                            st.error(tr("super_agent.step3.error", error=str(e)))
                            logger.exception(e)

            st.markdown(f"**{tr('super_agent.step3.clone_result')}**")
            audio_path = st.session_state.get("sa_audio_path")
            if audio_path and os.path.exists(str(audio_path)):
                st.audio(str(audio_path))
            else:
                st.caption(tr("super_agent.step3.clone_placeholder"))

            vol_col1, vol_col2 = st.columns(2)
            with vol_col1:
                st.slider(
                    tr("super_agent.step3.voice_volume"),
                    min_value=0.0, max_value=1.0, value=1.0, step=0.1,
                    key="sa_voice_volume",
                )
            with vol_col2:
                st.slider(
                    tr("super_agent.step3.bgm_volume"),
                    min_value=0.0, max_value=1.0, value=0.3, step=0.1,
                    key="sa_bgm_volume",
                )

    # ══════════════════════════════════════════════════════════════════
    # Step 4: 视频生成
    # ══════════════════════════════════════════════════════════════════
    def _render_step4_video(self, pixelle_video: Any):
        with st.container(border=True):
            st.markdown(f"**{tr('super_agent.step4.title')}**")

            char_col, upload_col = st.columns([3, 1])
            with char_col:
                character_options = {
                    "young_male": tr("super_agent.step4.character_young_male"),
                    "young_female": tr("super_agent.step4.character_young_female"),
                    "middle_male": tr("super_agent.step4.character_middle_male"),
                    "middle_female": tr("super_agent.step4.character_middle_female"),
                    "custom": tr("super_agent.step4.character_custom"),
                }
                character_key = st.selectbox(
                    tr("super_agent.step4.character_label"),
                    options=list(character_options.keys()),
                    format_func=lambda x: character_options[x],
                    key="sa_character_select",
                )
            with upload_col:
                st.markdown("<br>", unsafe_allow_html=True)
                uploaded_char = st.file_uploader(
                    tr("super_agent.step4.upload_character"),
                    type=["jpg", "jpeg", "png", "webp"],
                    key="sa_char_upload",
                    label_visibility="collapsed",
                )

            custom_char_path = None
            if uploaded_char:
                import uuid
                temp_dir = Path(f"temp/sa_char_{uuid.uuid4().hex[:8]}")
                temp_dir.mkdir(parents=True, exist_ok=True)
                custom_char_path = str(temp_dir / uploaded_char.name)
                with open(custom_char_path, "wb") as f:
                    f.write(uploaded_char.getbuffer())
                st.image(uploaded_char, width=120)

            audio_col, model_col = st.columns(2)
            with audio_col:
                driving_options = {
                    "auto": tr("super_agent.step4.driving_auto"),
                }
                st.selectbox(
                    tr("super_agent.step4.driving_audio"),
                    options=list(driving_options.keys()),
                    format_func=lambda x: driving_options[x],
                    key="sa_driving_audio",
                )
            with model_col:
                model_options = {
                    "v1": tr("super_agent.step4.model_v1"),
                    "v2": tr("super_agent.step4.model_v2"),
                }
                st.selectbox(
                    tr("super_agent.step4.model_version"),
                    options=list(model_options.keys()),
                    format_func=lambda x: model_options[x],
                    index=1,
                    key="sa_model_version",
                )

            if st.button(
                tr("super_agent.step4.generate_btn"),
                type="primary",
                width="stretch",
                key="sa_video_generate",
            ):
                audio_path = st.session_state.get("sa_audio_path")
                if not audio_path or not os.path.exists(str(audio_path)):
                    st.warning(tr("super_agent.step4.no_audio"))
                else:
                    self._do_generate_video(
                        pixelle_video,
                        audio_path=str(audio_path),
                        character_key=character_key,
                        custom_char_path=custom_char_path,
                        model_version=st.session_state.get("sa_model_version", "v2"),
                    )

            st.markdown(f"**{tr('super_agent.step4.video_result')}**")
            video_path = st.session_state.get("sa_video_path")
            if video_path and os.path.exists(str(video_path)):
                st.video(str(video_path))
            else:
                st.caption(tr("super_agent.step4.preview_placeholder"))

    # ══════════════════════════════════════════════════════════════════
    # Step 5: 一键成片
    # ══════════════════════════════════════════════════════════════════
    def _render_step5_assembly(self, pixelle_video: Any):
        with st.container(border=True):
            st.markdown(f"**{tr('super_agent.step5.title')}**")

            st.text_input(
                tr("super_agent.step5.video_source"),
                value=tr("super_agent.step5.video_source_auto"),
                disabled=True,
                key="sa_video_source_display",
            )

            st.markdown(f"**{tr('super_agent.step5.subtitle_settings')}**")
            sub_col1, sub_col2, sub_col3 = st.columns([1, 2, 1])
            with sub_col1:
                subtitle_enabled = st.toggle(
                    tr("super_agent.step5.subtitle_enable"),
                    value=True,
                    key="sa_subtitle_toggle",
                )
            with sub_col2:
                if subtitle_enabled:
                    st.button(
                        "A " + tr("super_agent.step5.subtitle_config"),
                        key="sa_subtitle_config",
                        width="stretch",
                    )
            with sub_col3:
                if subtitle_enabled:
                    st.button(
                        tr("super_agent.step5.subtitle_align"),
                        key="sa_subtitle_align",
                        width="stretch",
                    )

            if subtitle_enabled:
                st.checkbox(
                    tr("super_agent.step5.subtitle_no_template"),
                    value=True,
                    key="sa_subtitle_no_tpl",
                )

            tab_clip, tab_material, tab_bgm = st.tabs([
                tr("super_agent.step5.tab_clip"),
                tr("super_agent.step5.tab_material"),
                tr("super_agent.step5.tab_bgm"),
            ])

            with tab_clip:
                clip_col1, clip_col2 = st.columns(2)
                with clip_col1:
                    st.selectbox(
                        tr("super_agent.step5.tab_clip"),
                        [tr("super_agent.step5.clip_layout")],
                        key="sa_clip_layout",
                        label_visibility="collapsed",
                    )
                with clip_col2:
                    st.button(
                        tr("super_agent.step5.clip_select_bg"),
                        key="sa_clip_bg",
                        width="stretch",
                    )

            with tab_material:
                st.file_uploader(
                    tr("super_agent.step5.tab_material"),
                    type=["jpg", "jpeg", "png", "webp", "mp4", "mov"],
                    accept_multiple_files=True,
                    key="sa_extra_materials",
                    label_visibility="collapsed",
                )

            with tab_bgm:
                bgm_col1, bgm_col2 = st.columns([3, 1])
                with bgm_col1:
                    bgm_params = render_bgm_section(key_prefix="sa_")
                with bgm_col2:
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.button(
                        tr("super_agent.step5.ai_bgm"),
                        key="sa_ai_bgm",
                        width="stretch",
                    )

            st.toggle(
                tr("super_agent.step5.breath_enable"),
                value=False,
                key="sa_breath_removal",
            )

            edit_col, start_col = st.columns(2)
            with edit_col:
                st.button(
                    tr("super_agent.step5.edit_subtitle"),
                    width="stretch",
                    key="sa_edit_subtitle",
                )
            with start_col:
                assemble_clicked = st.button(
                    tr("super_agent.step5.start_btn"),
                    type="primary",
                    width="stretch",
                    key="sa_assemble_btn",
                )

            if assemble_clicked:
                video_path = st.session_state.get("sa_video_path")
                if not video_path or not os.path.exists(str(video_path)):
                    st.warning(tr("super_agent.step5.no_video"))
                else:
                    self._do_assemble(
                        pixelle_video,
                        video_path=str(video_path),
                        bgm_params=bgm_params,
                        subtitle_enabled=subtitle_enabled,
                    )

        st.markdown(f"**{tr('super_agent.step5.video_preview')}**")
        final_path = st.session_state.get("sa_final_video")
        if final_path and os.path.exists(str(final_path)):
            st.video(str(final_path))
            with open(str(final_path), "rb") as vf:
                st.download_button(
                    label="⬇️ 下载视频" if get_language() == "zh_CN" else "⬇️ Download Video",
                    data=vf.read(),
                    file_name=os.path.basename(str(final_path)),
                    mime="video/mp4",
                    width="stretch",
                )
        else:
            st.caption(tr("super_agent.step5.preview_placeholder"))

        render_version_info()

    # ══════════════════════════════════════════════════════════════════
    # Helpers
    # ══════════════════════════════════════════════════════════════════

    @staticmethod
    def _call_llm(pixelle_video: Any, prompt: str, target_key: str):
        """Call LLM via pixelle_video.llm() and store result in session state."""
        with st.spinner(tr("super_agent.step2.rewriting")):
            try:
                result = run_async(pixelle_video.llm(prompt, max_tokens=4096))
                st.session_state[target_key] = result
                st.rerun()
            except Exception as e:
                st.error(tr("super_agent.step2.rewrite_error", error=str(e)))
                logger.exception(e)

    def _do_generate_video(
        self,
        pixelle_video: Any,
        audio_path: str,
        character_key: str,
        custom_char_path: str | None,
        model_version: str = "v2",
    ):
        """Generate digital human video using ComfyKit workflows."""
        progress_bar = st.progress(0)
        status_text = st.empty()
        start_time = time.time()

        try:
            async def _generate():
                task_dir = st.session_state.get("sa_task_dir")
                if not task_dir:
                    task_dir, _ = create_task_output_dir()
                    st.session_state["sa_task_dir"] = task_dir

                kit = await pixelle_video._get_or_create_comfykit()

                character_image = custom_char_path
                if not character_image:
                    from pixelle_video.utils.os_util import get_resource_path, resource_exists
                    char_map = {
                        "young_male": "characters/young_male.png",
                        "young_female": "characters/young_female.png",
                        "middle_male": "characters/middle_male.png",
                        "middle_female": "characters/middle_female.png",
                    }
                    char_file = char_map.get(character_key, "characters/young_male.png")
                    if resource_exists("", char_file):
                        character_image = get_resource_path("", char_file)
                    else:
                        character_image = None

                status_text.text(tr("super_agent.step4.generating"))
                progress_bar.progress(20)

                wf_map = {
                    "v1": "workflows/runninghub/digital_combination.json",
                    "v2": "workflows/runninghub/digital_customize.json",
                }
                workflow_path = Path(wf_map.get(model_version, wf_map["v2"]))
                if not workflow_path.exists():
                    raise FileNotFoundError(
                        f"Digital human workflow not found: {workflow_path}"
                    )

                with open(workflow_path, "r", encoding="utf-8") as f:
                    wf_config = json.load(f)

                wf_params = {
                    "videoimage": character_image or "",
                    "audio": audio_path,
                }

                if wf_config.get("source") == "runninghub" and "workflow_id" in wf_config:
                    wf_input = wf_config["workflow_id"]
                else:
                    wf_input = str(workflow_path)

                progress_bar.progress(40)
                result = await kit.execute(wf_input, wf_params)

                logger.info(
                    f"Workflow result: status={getattr(result, 'status', 'N/A')}, "
                    f"msg={getattr(result, 'msg', 'N/A')}, "
                    f"videos={getattr(result, 'videos', None)}, "
                    f"videos_by_var={getattr(result, 'videos_by_var', None)}, "
                    f"images={getattr(result, 'images', None)}, "
                    f"images_by_var={getattr(result, 'images_by_var', None)}, "
                    f"audios={getattr(result, 'audios', None)}, "
                    f"audios_by_var={getattr(result, 'audios_by_var', None)}, "
                    f"texts={getattr(result, 'texts', None)}, "
                    f"outputs={getattr(result, 'outputs', None)}, "
                    f"duration={getattr(result, 'duration', None)}"
                )

                if getattr(result, "status", "") != "completed":
                    raise RuntimeError(
                        f"Workflow execution failed: "
                        f"status={getattr(result, 'status', 'N/A')}, "
                        f"msg={getattr(result, 'msg', 'N/A')}"
                    )

                generated_url = None
                for attr in ("videos", "videos_by_var", "images", "images_by_var"):
                    val = getattr(result, attr, None)
                    if val:
                        if isinstance(val, dict):
                            for k, v in val.items():
                                if v:
                                    generated_url = v[0] if isinstance(v, list) else v
                                    logger.info(f"Found media in result.{attr}[{k}]: {generated_url}")
                                    break
                        elif isinstance(val, list) and val:
                            generated_url = val[0]
                            logger.info(f"Found media in result.{attr}: {generated_url}")
                    if generated_url:
                        break

                if not generated_url and hasattr(result, "outputs") and result.outputs:
                    for node_id, node_out in result.outputs.items():
                        if isinstance(node_out, dict):
                            for key in ("videos", "gifs", "images"):
                                items = node_out.get(key)
                                if items:
                                    generated_url = items[0]
                                    logger.info(f"Found media in outputs[{node_id}][{key}]: {generated_url}")
                                    break
                        if generated_url:
                            break

                if not generated_url:
                    raise RuntimeError(
                        f"Workflow completed but no media found in result. "
                        f"Dump: {result.model_dump() if hasattr(result, 'model_dump') else str(result)}"
                    )

                progress_bar.progress(80)
                final_path = os.path.join(task_dir, "digital_human.mp4")
                timeout = httpx.Timeout(300.0)
                async with httpx.AsyncClient(timeout=timeout) as client:
                    resp = await client.get(generated_url)
                    resp.raise_for_status()
                    with open(final_path, "wb") as f:
                        f.write(resp.content)

                progress_bar.progress(100)
                return final_path

            video_path = run_async(_generate())
            status_text.text(tr("super_agent.step4.success"))
            st.session_state["sa_video_path"] = video_path

            total_time = time.time() - start_time
            if os.path.exists(video_path):
                size_mb = os.path.getsize(video_path) / (1024 * 1024)
                st.caption(
                    f"⏱️ {tr('info.generation_time')} {total_time:.1f}s   "
                    f"📦 {size_mb:.2f}MB"
                )
            st.rerun()

        except Exception as e:
            status_text.text("")
            progress_bar.empty()
            st.error(tr("super_agent.step4.error", error=str(e)))
            logger.exception(e)

    def _do_assemble(
        self,
        pixelle_video: Any,
        video_path: str,
        bgm_params: dict,
        subtitle_enabled: bool,
    ):
        """Assemble final video with BGM, subtitles, etc."""
        progress_bar = st.progress(0)
        status_text = st.empty()
        start_time = time.time()

        try:
            task_dir = st.session_state.get("sa_task_dir")
            if not task_dir:
                task_dir, _ = create_task_output_dir()
                st.session_state["sa_task_dir"] = task_dir

            status_text.text(tr("super_agent.step5.assembling"))
            progress_bar.progress(20)

            final_path = os.path.join(task_dir, "final.mp4")
            bgm_path = bgm_params.get("bgm_path")
            bgm_volume = bgm_params.get("bgm_volume", 0.2)

            from pixelle_video.services.video import VideoService
            video_svc = VideoService({})

            resolved_bgm = None
            if bgm_path:
                from pixelle_video.utils.os_util import get_resource_path, resource_exists
                if resource_exists("bgm", bgm_path):
                    resolved_bgm = get_resource_path("bgm", bgm_path)

            progress_bar.progress(50)

            video_svc.concat_videos(
                videos=[video_path],
                output=final_path,
                bgm_path=resolved_bgm,
                bgm_volume=bgm_volume,
            )

            progress_bar.progress(100)
            status_text.text(tr("super_agent.step5.success"))
            st.session_state["sa_final_video"] = final_path

            total_time = time.time() - start_time
            if os.path.exists(final_path):
                size_mb = os.path.getsize(final_path) / (1024 * 1024)
                st.caption(
                    f"⏱️ {tr('info.generation_time')} {total_time:.1f}s   "
                    f"📦 {size_mb:.2f}MB"
                )
            st.rerun()

        except Exception as e:
            status_text.text("")
            progress_bar.empty()
            st.error(tr("super_agent.step5.error", error=str(e)))
            logger.exception(e)


register_pipeline_ui(SuperAgentPipelineUI)
