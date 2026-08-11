import json
import os
import random
import re
import tempfile

import gradio as gr

from ollama import Client


def read_prompt(prompt_file):
    with open(prompt_file, "r", encoding="utf-8") as md_file:
        prompt = md_file.read()
    return prompt


def generate_response(instruction):
    try:
        result = ollama_client.generate(
            model=model_name,
            prompt=instruction,
            system=system_prompt,
            stream=False,
            options={
                "temperature": 1.0,
                "top_p": 0.95,
                "top_k": 50,
                "num_ctx": 2048,
                "repetition_penalty": 1.05
            }
        )

        model_response = result["response"]

        # 提取json内容
        json_match = re.search(r'```json(.*?)```', model_response, re.DOTALL)
        json_content = json_match.group(1).strip() if json_match else None

        if json_content is None:
            raise Exception("json_content is not valid")

        json_content = json.loads(json_content)

        return model_response, json_content
    except Exception as e:
        print(f"模型问答出错，重新回复中: {e}")
        generate_response(instruction)


def model_chat(production_introduce, history):
    history.clear()
    total_time = random.randint(25, 40)

    instruction = (f"请基于以下产品介绍，创作一个短视频带货脚本，时长{total_time}s左右。脚本需包含多个镜头，详细描述每个镜头的time_axis（时间轴）、"
                   "shot_type（镜头类型）、visual_description（画面描述）、voiceover（口播）和music（背景音乐）。"
                   "\n产品介绍：") + production_introduce
    print(f"用户指令:\n{instruction}")
    model_response, json_content = generate_response(instruction)
    print(f"模型回复:\n{model_response}")

    return model_response, json_content


def create_download_file(json_data):
    json_string = json.dumps(json_data, indent=4, ensure_ascii=False)

    with tempfile.NamedTemporaryFile(mode='w', suffix=".json", delete=False, encoding='utf-8') as temp_f:
        temp_f.write(json_string)
        temp_filepath = temp_f.name

    return temp_filepath


if __name__ == "__main__":
    # 模型名称
    model_name = "gemma_3_12B_q80:latest"
    ollama_host_url = "http://127.0.0.1:11434"

    # 创建模型客户端
    ollama_client = Client(
        host=ollama_host_url
    )

    # 系统提示
    system_prompt = read_prompt(os.path.join(os.path.dirname(__file__), "system_prompt.md"))

    # 界面
    with gr.Blocks(fill_width=True, theme="soft", title="分镜脚本生成助手", fill_height=True) as demo:
        json_comp = gr.JSON(value={"message": "无信息！"}, height=713, max_height=1000, label="脚本生成结果",
                            render=False)
        with gr.Row():
            with gr.Column():
                gr.Markdown("## 产品介绍输入")
                gr.Markdown("产品介绍，可以包括产品名称、用户痛点、产品卖点、适用场景、促销活动等信息")
                model_input = gr.Textbox(
                    placeholder="请输入具有实际意义的产品介绍...",
                    submit_btn=True,
                    lines=4
                )
                model_message = gr.Chatbot(
                    height=645,
                    max_height=1000,
                    type="messages",
                    label="分镜脚本生成助手"
                )
                gr.ChatInterface(
                    fn=model_chat,
                    type="messages",
                    theme="soft",
                    chatbot=model_message,
                    textbox=model_input,
                    additional_outputs=[json_comp]
                )
            with gr.Column():
                gr.Markdown("## 分镜脚本输出")
                gr.Markdown("分镜脚本，包括时间轴、镜头类型、画面描述、口播文案、背景音乐")
                json_comp.render()
                download_button = gr.DownloadButton(
                    "下载分镜脚本",
                    value=lambda x: create_download_file(x),
                    inputs=json_comp,
                    variant="primary"
                )
    # 启动
    demo.queue().launch(server_port=9999)
