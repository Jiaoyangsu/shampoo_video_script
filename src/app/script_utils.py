import json
import tempfile
from ollama import Client
import random
import re


# 下载JSON文件的函数
def create_download_file(json_data):
    json_string = json.dumps(json_data, indent=4, ensure_ascii=False)

    with tempfile.NamedTemporaryFile(mode='w', suffix=".json", delete=False, encoding='utf-8') as temp_f:
        temp_f.write(json_string)
        temp_filepath = temp_f.name

    return temp_filepath


# 读取提示文件的函数
def read_prompt(prompt_file):
    with open(prompt_file, "r", encoding="utf-8") as md_file:
        prompt = md_file.read()
    return prompt


# 重置所有输入框的函数
def reset_fields():
    return "", "", "", "", "", "", {"message": "尚未生成脚本！"}


# 生成JSON响应的函数
def generate_response(product_name, product_features, target_audience, user_pain_points, usage_scenarios, promotions, retry_count=0):
    # 添加输入验证
    errors = []
    if not product_name or not product_name.strip():
        errors.append("产品名称不能为空")
    if not product_features or not product_features.strip():
        errors.append("产品买点不能为空")
    if not target_audience or not target_audience.strip():
        errors.append("适用人群不能为空")
    if not user_pain_points or not user_pain_points.strip():
        errors.append("用户痛点不能为空")
    if not usage_scenarios or not usage_scenarios.strip():
        errors.append("使用场景不能为空")

    # 如果有错误，返回错误信息
    if errors:
        return False, {"error": "输入验证失败", "details": errors}

    # 模型名称
    model_name = "gemma_3_12B_q80:latest"
    ollama_host_url = "http://127.0.0.1:11434"

    # 创建模型客户端
    ollama_client = Client(
        host=ollama_host_url
    )

    system_prompt = read_prompt("./system_prompt.md")

    total_time = random.randint(25, 40)

    instruction = f"""请基于以下产品介绍，创作一个带货短视频分镜脚本，时长{total_time}s左右。产品介绍：\n1.产品名称：{product_name}\n2.产品卖点：{product_features}\n3.适用人群：{target_audience}\n4.用户痛点：{user_pain_points}\n5.使用场景：{usage_scenarios}\n6.促销活动：{promotions}"""

    print("instruction: \n", instruction)

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
        print("json_content: \n", json_content)

        return True, json_content
    except Exception as e:
        print(f"模型问答出错，重新回复中: {e}")
        # 检查是否达到最大重试次数
        if retry_count >= 4:
            return False, {"error": "模型请求失败", "details": f"已达到最大重试次数(5次)，最后一次错误: {str(e)}"}
        # 增加重试计数并重试
        return generate_response(product_name, product_features, target_audience, user_pain_points, usage_scenarios, promotions, retry_count + 1)
