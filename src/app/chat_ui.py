"""Gradio app for the shampoo script generator.

The app is now a factory (``create_demo``) instead of a module-level ``demo``
created only under ``if __name__ == "__main__"``.  That was why
``scripts/run_app.py`` could never import it.
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Optional, Tuple

import gradio as gr

from src.finetune.prompting import build_user_content
from src.inference.backends import (
    Backend,
    BackendConfigError,
    build_backend_from_env,
    read_system_prompt,
)
from src.script_schema import (
    ScriptValidationError,
    extract_json,
    validate_script,
    validation_markdown,
)

DEFAULT_MAX_NEW_TOKENS = int(os.environ.get("SCRIPT_MAX_NEW_TOKENS", "700"))


def _instruction(description: str, duration: int) -> str:
    # Same shape as SFT/DPO training: target duration/shots + product brief.
    return build_user_content(description, duration=duration, shots="10–12")


def _parse(raw: str) -> Tuple[Optional[Any], str]:
    try:
        parsed = extract_json(raw)
    except (ScriptValidationError, ValueError, TypeError) as exc:
        return None, "### 自检：❌ 未通过\n\nJSON 解析失败：`%s`" % exc
    validation = validate_script(
        parsed,
        min_shots=1,
        require_music=True,
        enforce_pacing=False,
    )
    return parsed, validation_markdown(validation)


def create_download_file(json_data: Any) -> str:
    json_string = json.dumps(json_data, indent=2, ensure_ascii=False)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as temp_file:
        temp_file.write(json_string)
        return temp_file.name


def build_app(
    backend: Optional[Backend] = None,
    *,
    system_prompt: Optional[str] = None,
    backend_error: Optional[str] = None,
    max_attempts: int = 2,
) -> gr.Blocks:
    """Build the Gradio UI.  Importable without starting a server."""
    prompt_text = system_prompt or read_system_prompt()
    if backend is None and backend_error is None:
        try:
            backend = build_backend_from_env()
        except BackendConfigError as exc:
            backend_error = str(exc)

    def generate(description: str, duration: int):
        if backend is None:
            return (
                {"error": backend_error or "backend not configured"},
                "### 自检：❌ 未通过\n\n%s" % (backend_error or "backend not configured"),
                "",
            )
        if not description or not description.strip():
            return (
                {"error": "产品介绍不能为空"},
                "### 自检：❌ 未通过\n\n产品介绍不能为空。",
                "",
            )

        raw = ""
        parsed = None
        validation_text = ""
        for attempt in range(1, max_attempts + 1):
            try:
                result = backend.generate(
                    prompt_text,
                    _instruction(description, duration),
                    max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
                    temperature=0.7,
                    top_p=0.9,
                    seed=42 + attempt,
                )
            except Exception as exc:  # surface a useful error in the UI
                return (
                    {"error": "%s: %s" % (type(exc).__name__, exc)},
                    "### 自检：❌ 未通过\n\n模型调用失败：`%s`" % exc,
                    raw,
                )
            raw = result.text
            parsed, validation_text = _parse(raw)
            if parsed is not None:
                break

        if parsed is None:
            return (
                {"error": "JSON 解析失败", "raw": raw},
                validation_text
                + "\n\n> 已重试 %d 次；请检查模型是否被截断，或增大 max_new_tokens。"
                % max_attempts,
                raw,
            )
        return parsed, validation_text, raw

    with gr.Blocks(
        title="洗发水带货分镜脚本生成器",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            "## 洗发水带货分镜脚本生成器\n"
            "输入产品介绍，生成包含时间轴、镜头类型、画面描述、口播和背景音乐的 JSON 分镜脚本。"
        )
        if backend_error:
            gr.Markdown(
                "> ⚠️ 模型后端未配置：`%s`\n>\n"
                "> 本地演示可设置 `OLLAMA_MODEL`；服务器上可设置 `SCRIPT_BACKEND=transformers` "
                "和 `SCRIPT_BASE_MODEL` / `SCRIPT_SFT_ADAPTER` / `SCRIPT_DPO_ADAPTER`。"
                % backend_error
            )
        with gr.Row():
            with gr.Column(scale=3):
                description = gr.Textbox(
                    label="产品介绍",
                    lines=8,
                    placeholder=(
                        "例如：产品名称、核心卖点、目标人群、用户痛点、"
                        "使用场景、促销信息……"
                    ),
                )
                duration = gr.Slider(
                    minimum=15,
                    maximum=60,
                    value=30,
                    step=5,
                    label="目标时长（秒）",
                )
                generate_btn = gr.Button("生成分镜脚本", variant="primary")
            with gr.Column(scale=4):
                json_output = gr.JSON(label="分镜脚本 JSON", height=520)
                check_output = gr.Markdown("### 自检\n\n等待生成……")
                download_btn = gr.DownloadButton(
                    "下载 JSON", variant="secondary"
                )
        with gr.Accordion("模型原始输出", open=False):
            raw_output = gr.Textbox(lines=12, label="raw", interactive=False)

        generate_btn.click(
            fn=generate,
            inputs=[description, duration],
            outputs=[json_output, check_output, raw_output],
        )
        download_btn.click(
            fn=create_download_file,
            inputs=[json_output],
            outputs=[download_btn],
        )
    return demo


def main() -> None:
    demo = build_app()
    demo.queue().launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "9999")),
        share=os.environ.get("GRADIO_SHARE", "0") == "1",
    )


if __name__ == "__main__":
    main()
