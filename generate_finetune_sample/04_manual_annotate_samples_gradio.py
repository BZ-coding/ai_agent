import json
import os
import random
import sys

import gradio as gr
import pandas as pd

print(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.ai_agent import AIAgent
from utils.chatbot import ChatBot

FORMATED_SAMPLES_PATH = 'checked_finetune_sample_formatted.jsonl'

if gr.NO_RELOAD:
    chatbot = ChatBot()
    ai_agent = AIAgent(chatbot=chatbot)

# 从JSONL文件逐行读取数据
dir_path = os.path.dirname(os.path.realpath(__file__))
datas = []
with open(os.path.join(dir_path, FORMATED_SAMPLES_PATH), 'r') as f:
    for line in f:
        data = json.loads(line)
        datas.append(data)
current_data: dict = None
current_messages = None


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


def start_conversation():
    global current_data, current_messages
    current_data = get_no_verify_data()
    current_messages = []

    return continue_conversation(pd.DataFrame())


def save_func(dataframe):
    global current_data
    convert_dataframe_to_current_message(dataframe=dataframe)
    yield convert_message_to_dataframe(current_messages)

    current_data["messages"] = current_messages
    current_data["is_verify"] = True
    save_datas()


def delete_func():
    global current_data, current_messages
    delete_data(id=current_data['id'])
    current_data = None
    current_messages = None
    save_datas()
    return []


def convert_message_to_dataframe(messages):
    messages_ = []
    for message in messages:  # After edit, the \n will miss
        messages_.append({"role": message["role"], "content": message["content"].replace("\n", "\n\u0000")})
    return pd.DataFrame.from_records(messages_)


def convert_dataframe_to_current_message(dataframe):
    global current_messages
    messages = dataframe.to_dict(orient='records')
    result = []
    for message in messages:
        content = message['content']
        if '\n' not in content:
            content = content.replace("\u0000", "\n")
        else:
            content = content.replace("\u0000", "")  # don't edit cell will have \u0000
        result.append({"role": message["role"], "content": content})
    current_messages = result


def continue_conversation(dataframe):
    global current_messages, current_messages
    convert_dataframe_to_current_message(dataframe)
    print("continue_conversation : ", current_messages)
    query = current_data['query']
    for response_messages in ai_agent.ai_agent_chat(query=query, temperature=0.0, is_print=True, messages=current_messages):
        current_messages = response_messages
        return convert_message_to_dataframe(current_messages)


with gr.Blocks() as demo:
    gr.Markdown("# Manually Annotate Samples")
    outputs = gr.DataFrame(headers=["role", "content"],
                           datatype=["str", "markdown"],
                           interactive=True,
                           wrap=True)
    with gr.Row():
        start_button = gr.Button("Start & Next")
        continue_button = gr.Button("Continue conversation")
        save_button = gr.Button("Save")
        del_button = gr.Button("Delete")
    start_button.click(fn=start_conversation, outputs=[outputs])
    save_button.click(fn=save_func, inputs=[outputs], outputs=[outputs])
    del_button.click(fn=delete_func, outputs=[outputs])
    continue_button.click(fn=continue_conversation, inputs=[outputs], outputs=[outputs])

    # commit_btn.click(fn=add_text, inputs=[chatbot, txt], outputs=[chatbot, txt]).then(
    #     bot, chatbot, chatbot, api_name="bot_response"
    # ).then(lambda: gr.Textbox(interactive=True), None, [txt], queue=False)

demo.launch(server_name="0.0.0.0",
            share=False,
            debug=True,
            )

# https://blog.csdn.net/jclian91/article/details/132417892
