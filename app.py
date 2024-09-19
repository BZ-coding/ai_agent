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
    chatbot = ChatBot(model_name="zhangsheng377/llama-3-chinese-8b-tools-f16-lora")
    ai_agent = AIAgent(chatbot=chatbot)
query = None
current_messages = None


def start_conversation(query_textbox):
    global current_messages, query
    current_messages = []
    query = query_textbox
    return continue_conversation(pd.DataFrame())


def convert_message_to_dataframe(messages):
    messages_ = []
    id = 0
    for message in messages:  # After edit, the \n will miss
        messages_.append({"id": id, "role": message["role"], "content": message["content"]})
        id += 1
    return pd.DataFrame.from_records(messages_)


def convert_dataframe_to_current_message(dataframe):
    global current_messages
    messages = dataframe.to_dict(orient='records')
    result = []
    for message in messages:
        content = message['content']
        result.append({"role": message["role"], "content": content})
    current_messages = result


def continue_conversation(dataframe):
    global current_messages, query
    convert_dataframe_to_current_message(dataframe)
    print("continue_conversation : ", current_messages)
    skip_chatbot = (len(current_messages) == 0 or
                    (current_messages[-1]["content"].endswith("Observation:") or
                     current_messages[-1]["content"].endswith("Observation:\n")))
    for response_messages in ai_agent.ai_agent_chat(query=query, temperature=0.0, is_print=True,
                                                    messages=current_messages, skip_chatbot=skip_chatbot):
        current_messages = response_messages
        return convert_message_to_dataframe(current_messages)


def select_conversation(index):
    try:
        id = int(index)
    except:
        return ""
    if id not in range(len(current_messages)):
        return ""
    return current_messages[id]["content"]


def change_conversation(index, content):
    global current_messages
    try:
        id = int(index)
    except:
        return convert_message_to_dataframe(current_messages)
    if id not in range(len(current_messages)):
        return convert_message_to_dataframe(current_messages)
    current_messages[id]["content"] = content
    if not content:
        current_messages.pop(id)
    return convert_message_to_dataframe(current_messages)


with gr.Blocks() as demo:
    gr.Markdown("# Manually Annotate Samples")
    with gr.Row():
        with gr.Column():
            query_textbox = gr.Textbox(label="query", interactive=True)
            edit_index_textbox = gr.Textbox(label="Edit index")
            edit_content_textbox = gr.Textbox(label="Edit content", interactive=True)
        outputs = gr.DataFrame(headers=["id", "role", "content"],
                               datatype=["number", "str", "markdown"],
                               interactive=False,
                               wrap=True)
    with gr.Row():
        start_button = gr.Button("Start & Next")
        continue_button = gr.Button("Continue conversation")
    start_button.click(fn=start_conversation, inputs=[query_textbox], outputs=[outputs])
    continue_button.click(fn=continue_conversation, inputs=[outputs], outputs=[outputs])
    edit_index_textbox.input(fn=select_conversation, inputs=[edit_index_textbox], outputs=[edit_content_textbox])
    edit_content_textbox.change(fn=change_conversation, inputs=[edit_index_textbox, edit_content_textbox],
                                outputs=[outputs])

demo.launch(server_name="0.0.0.0",
            share=True,
            debug=False,
            )
