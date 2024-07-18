import json
import os
import random

import gradio as gr
from openai import OpenAI

FORMATED_SAMPLES_PATH = 'checked_finetune_sample_formatted.jsonl'
OPENAI_BASE_URL = 'http://localhost:11434/v1/'
MODEL_NAME = 'llama-3-chinese-8b-instruct-v3-f16'

if gr.NO_RELOAD:
    client = OpenAI(
        base_url=OPENAI_BASE_URL,
        api_key='ollama',  # required but ignored
    )

# 从JSONL文件逐行读取数据
dir_path = os.path.dirname(os.path.realpath(__file__))
datas = []
with open(os.path.join(dir_path, FORMATED_SAMPLES_PATH), 'r') as f:
    for line in f:
        data = json.loads(line)
        datas.append(data)
current_data: dict = None
current_messages = None


def get_response(**params):
    chat_completion = client.chat.completions.create(
        **params,
        model=MODEL_NAME,
    )
    return chat_completion


def run_conversation(query: str, messages: list, stream=False):
    params = {
        "messages": messages,
        "temperature": 0.6,
        "stream": stream,
    }
    response = get_response(**params)
    return response


def get_no_verify_data():
    for data in datas:
        if data['is_verify']:
            continue
        return data
    raise RuntimeError('No valid data')


def delete_data(id):
    for i in range(len(datas)):
        data = datas[i]
        if data['id'] == id:
            datas.pop(i)
            return
    raise RuntimeError(f'Not find data id:{id}')


def save_datas():
    with open(os.path.join(dir_path, FORMATED_SAMPLES_PATH), 'w') as f:
        for data in datas:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')


def chatbot_interface():
    global current_data, current_messages
    current_data = get_no_verify_data()
    query = current_data['query']
    current_messages = [
        {"role": "user", "content": query}
    ]

    stream = True
    response = run_conversation(query, messages=current_messages, stream=stream)
    if not stream:
        return response.choices[0].message.content
    message = ""
    for token in response:
        if token.choices[0].finish_reason is not None:
            continue
        message += token.choices[0].delta.content
        yield query, message


def save_func():
    global current_data, current_messages
    current_data["messages"] = current_messages
    current_data["is_verify"] = True
    save_datas()


def delete_func():
    global current_data, current_messages
    delete_data(id=current_data['id'])
    current_data = None
    current_messages = None
    save_datas()
    return "", ""


with gr.Blocks() as demo:
    gr.Markdown("# Manually Annotate Samples")
    with gr.Row():
        query_textbox = gr.Textbox(label="Query", interactive=False),
        response_textbox = gr.Textbox(label="Response", interactive=True)
    with gr.Row():
        start_button = gr.Button("Start & Next")
        del_button = gr.Button("Delete")
        save_button = gr.Button("Save")
    start_button.click(fn=chatbot_interface, outputs=[query_textbox[0], response_textbox])
    save_button.click(fn=save_func)
    del_button.click(fn=delete_func, outputs=[query_textbox[0], response_textbox])

demo.launch(server_name="0.0.0.0",
            share=False,
            debug=True,
            )
