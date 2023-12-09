import os
import streamlit as st
import torch
import requests
import json
import requests
import json
import urllib.request
import base64
import os
from PIL import Image
import numpy as np

base_url = "http://127.0.0.1:8000"
out_dir = 'txt2img'
os.makedirs(out_dir, exist_ok=True)
webui_server_url = 'http://127.0.0.1:7860'

def call_api(api_endpoint, **payload):
    data = json.dumps(payload).encode('utf-8')
    request = urllib.request.Request(
        f'{webui_server_url}/{api_endpoint}',
        headers={'Content-Type': 'application/json'},
        data=data,
    )
    response = urllib.request.urlopen(request)
    return json.loads(response.read().decode('utf-8'))

def decode_and_save_base64(base64_str, save_path):
    with open(save_path, "wb") as file:
        file.write(base64.b64decode(base64_str))

def call_txt2img_api(**payload):
    response = call_api('sdapi/v1/txt2img', **payload)
    for index, image in enumerate(response.get('images')):
        save_path = os.path.join(out_dir, f'txt2img.png')
        decode_and_save_base64(image, save_path)

def create_chat_completion(model, messages, message_placeholder, max_length, top_p, temperature, history, functions=None, use_stream=True):
    data = {
        "function": functions,  # 函数定义
        "model": model,  # 模型名称
        "messages": messages,  # 会话历史
        "stream": use_stream,  # 是否流式响应
        "max_tokens": max_length,  # 最多生成字数
        "temperature": temperature,  # 温度
        "top_p": top_p,  # 采样概率
    }

    response = requests.post(f"{base_url}/v1/chat/completions", json=data, stream=use_stream)
    if response.status_code == 200:
        if use_stream:
            # 处理流式响应
            output = ''
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')[6:]
                    try:
                        response_json = json.loads(decoded_line)
                        content = response_json.get("choices", [{}])[0].get("delta", {}).get("content", "")
                        output += content
                        message_placeholder.markdown(output)
                    except:
                        print("Special Token:", decoded_line)
            history.append({'role': 'assistant', 'metadata': '', 'content': output})
            payload = {
                "prompt": output,  # extra networks also in prompts
                "negative_prompt": "",
                "seed": 1,
                "steps": 20,
                "width": 512,
                "height": 512,
                "cfg_scale": 7,
                "sampler_name": "DPM++ 2M Karras",
                "n_iter": 1,
                "batch_size": 1,
            }
            call_txt2img_api(**payload)
            st.image('txt2img/txt2img.png')
        else:
            # 处理非流式响应
            decoded_line = response.json()
            content = decoded_line.get("choices", [{}])[0].get("message", "").get("content", "")
            print(content)
    else:
        print("Error:", response.status_code)
        return None

# 设置页面标题、图标和布局
st.set_page_config(
    page_title="Chat Interactive Game",
    page_icon=":robot:",
    layout="wide"
)

# 初始化历史记录和past key values
if "history" not in st.session_state:
    st.session_state.history = []
if "past_key_values" not in st.session_state:
    st.session_state.past_key_values = None

# 设置max_length、top_p和temperature
max_length = st.sidebar.slider("max_length", 0, 32768, 8192, step=1)
top_p = st.sidebar.slider("top_p", 0.0, 1.0, 0.8, step=0.01)
temperature = st.sidebar.slider("temperature", 0.0, 1.0, 0.6, step=0.01)

# 清理会话历史
buttonClean = st.sidebar.button("清理会话历史", key="clean")
if buttonClean:
    st.session_state.history = []
    st.session_state.past_key_values = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    st.rerun()

# 渲染聊天历史记录
for i, message in enumerate(st.session_state.history):
    if message["role"] == "user":
        with st.chat_message(name="user", avatar="user"):
            st.markdown(message["content"])
    else:
        with st.chat_message(name="assistant", avatar="assistant"):
            st.markdown(message["content"])

# 输入框和输出框
with st.chat_message(name="user", avatar="user"):
    input_placeholder = st.empty()
with st.chat_message(name="assistant", avatar="assistant"):
    message_placeholder = st.empty()

# 获取用户输入
prompt_text = st.chat_input("请输入您的问题")

# 如果用户输入了内容,则生成回复
if prompt_text:

    input_placeholder.markdown(prompt_text)
    history = st.session_state.history
    past_key_values = st.session_state.past_key_values
    chat_messages = st.session_state.history
    chat_messages.append({"role": "user","content": prompt_text})
    create_chat_completion(
        "chatglm3-6b",
        messages=chat_messages,
        message_placeholder=message_placeholder,
        max_length=max_length,
        top_p=top_p,
        temperature=temperature,
        history=history
    )

    # 更新历史记录
    st.session_state.history = history
