import os
import time
import base64
from openai import OpenAI

def encode_image(image_path):
    """Encode image to base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Start timer
start_time = time.time()

# Local image path - change this to your image file
image_path = "1.png"  # Change this to your local image path

# Check if image exists
if not os.path.exists(image_path):
    print(f"Error: Image file '{image_path}' not found!")
    print("Please update the image_path variable with a valid local image file.")
    exit(1)

print(f"Using local image: {image_path}")

client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    api_key="sk-327233dd8f1f4012a7b25283b5da673d",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# Timer for API initialization
init_time = time.time()
print(f"Client initialization took: {init_time - start_time:.3f} seconds")

# Timer for image encoding
encode_start = time.time()
base64_image = encode_image(image_path)
encode_end = time.time()
print(f"Image encoding took: {encode_end - encode_start:.3f} seconds")

# Timer for API call
api_start = time.time()
completion = client.chat.completions.create(
    model="qwen-vl-max",  # 此处以qwen-vl-plus为例，可按需更换模型名称。模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
    messages=[{"role": "user","content": [
            {"type": "image_url",
             "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
            {"type": "text", "text": "在这里人的心情怎么样,用一字"},
            ]}]
    )
api_end = time.time()
print(f"API call took: {api_end - api_start:.3f} seconds")

print(completion.model_dump_json())

# Total execution time
end_time = time.time()
total_time = end_time - start_time
print(f"\nTotal execution time: {total_time:.3f} seconds")