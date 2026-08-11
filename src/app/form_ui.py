import gradio as gr
import json
import tempfile
from ollama import Client
import random
import re

# 读取提示文件的函数
def read_prompt(prompt_file):
    with open(prompt_file, "r", encoding="utf-8") as md_file:
        prompt = md_file.read()
    return prompt

# 重置所有输入框的函数
def reset_fields():
    return "", "", "", "", "", "", {"message": "尚未生成脚本！"}

# 生成JSON响应的函数
def generate_response(product_name, product_features, target_audience, user_pain_points, usage_scenarios, promotions):
    total_time = random.randint(25, 40)
    
    instruction=f"""请基于以下产品介绍，创作一个带货短视频分镜脚本，时长{total_time}s左右。产品介绍：\n
    1.产品名称：{product_name}\n
    2.产品卖点：{product_features}\n
    3.适用人群：{target_audience}\n
    4.用户痛点：{user_pain_points}\n
    5.使用场景：{usage_scenarios}\n
    6.促销活动：{promotions}"""

    print("instruction: \n",instruction)

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
        gr.Info("脚本生成成功！")
        print("json_content: \n",json_content)

        return json_content
    except Exception as e:
        print(f"模型问答出错，重新回复中: {e}")
        generate_response(product_name, product_features, target_audience, user_pain_points, usage_scenarios, promotions)

# 下载JSON文件的函数
def create_download_file(json_data):
    json_string = json.dumps(json_data, indent=4, ensure_ascii=False)

    with tempfile.NamedTemporaryFile(mode='w', suffix=".json", delete=False, encoding='utf-8') as temp_f:
        temp_f.write(json_string)
        temp_filepath = temp_f.name

    return temp_filepath

# 构建Gradio界面
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

    # 软件界面
    with gr.Blocks(title="视频分镜脚本生成", theme=gr.themes.Soft(), css=".row-container {display: flex; flex-wrap: nowrap; width: 70%; margin: 0 auto;} #left-column, #right-column {border: 1px solid #ccc; border-radius: 0.5rem; padding: 0.8rem 0.5rem;}") as demo:
        
        with gr.Row(elem_classes="row-container"):
            # 左侧输入区域
            with gr.Column(scale=1, elem_id="left-column"):
                gr.Markdown("## 产品信息")
                
                product_name = gr.Textbox(label="产品名称", placeholder="请输入产品名称")
                product_features = gr.Textbox(label="产品买点", placeholder="请输入产品卖点，多个卖点使用分号分隔", lines=3)
                target_audience = gr.Textbox(label="适用人群", placeholder="产品适用的人群，例如：'青少年'")
                user_pain_points = gr.Textbox(label="用户痛点", placeholder="产品能解决用户的什么痛点，比如：'价格昂贵'")
                usage_scenarios = gr.Textbox(label="使用场景", placeholder="产品的使用场景，例如：'家庭使用'")
                promotions = gr.Textbox(label="促销活动", placeholder="请输入价格/优惠活动，例如：'八折优惠'", lines=3)
                
                with gr.Row():
                    reset_btn = gr.Button("重置", variant="secondary",scale=2)
                    generate_btn = gr.Button("生成", variant="primary",scale=8)
                
            # 右侧预览区域
            with gr.Column(scale=1, elem_id="right-column"):
                gr.Markdown("## 脚本预览")
                json_preview = gr.JSON(value={"message": "尚未生成脚本！"}, height=650, max_height=1000, label="生成结果")
                with gr.Row():
                    download_btn = gr.DownloadButton("下载", variant="secondary")
        
        # 设置按钮的点击事件
        generate_btn.click(
            fn=generate_response,
            inputs=[product_name, product_features, target_audience, user_pain_points, usage_scenarios, promotions],
            outputs=[json_preview]
        )
        
        reset_btn.click(
            fn=reset_fields,
            inputs=[],
            outputs=[product_name, product_features, target_audience, user_pain_points, usage_scenarios, promotions, json_preview]
        )

        download_btn.click(
            fn=create_download_file,
            inputs=[json_preview],
            outputs=[download_btn]
        )
    
    # 启动应用
    demo.queue().launch(server_port=7861, share=False, debug=True)