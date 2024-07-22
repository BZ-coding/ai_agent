import json
import os
import random

import gradio as gr
import pandas as pd
from openai import OpenAI

FORMATED_SAMPLES_PATH = 'checked_finetune_sample_formatted.jsonl'
OPENAI_BASE_URL = 'http://localhost:11434/v1/'
MODEL_NAME = 'llama-3-chinese-8b-instruct-v3-f16'
# MODEL_NAME = 'qwen2:7b'
STEAM = True

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


def run_conversation(messages: list, stream=False):
    response = client.chat.completions.create(
        messages=messages,
        temperature=0.6,
        stream=stream,
        model=MODEL_NAME,
        stop=['Observation:', 'Observation:\n'],
    )
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
    current_messages = [  # TODO: wait to use ai_agent class
        {"role": "system",
         "content": "Answer the following questions as best you can. You have access to the following tools:\n\nCalculator(*args: Any, callbacks: Union[List[langchain_core.callbacks.base.BaseCallbackHandler], langchain_core.callbacks.base.BaseCallbackManager, NoneType] = None, tags: Optional[List[str]] = None, metadata: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any - Useful for when you need to answer questions about math.\nSearch(query: str) -> str - A wrapper around Search. Useful for when you need to answer questions about current events. Input should be a search query.\n\nUse the following format:\n\nQuestion: the input question you must answer\nThought: you should always think about what to do\nAction: the action to take, should be one of [Calculator, Search]\nAction Input: the input to the action\nObservation: the result of the action\n... (this Thought/Action/Action Input/Observation can repeat N times)\nThought: I now know the final answer\nFinal Answer: the final answer to the original input question\n\nBegin!\n\n"},
        {"role": "user", "content": f"Question: {query}\n"},
        {"role": "assistant", "content": f"Thought: "},
    ]

    for response_dataframe in get_response_and_dataframe(messages=current_messages, stream=STEAM):
        yield response_dataframe


def get_response(messages, stream):
    response = run_conversation(messages=messages, stream=stream)
    if not stream:
        message = response.choices[0].message.content
        yield message
    else:
        for token in response:
            if token.choices[0].finish_reason is not None:
                continue
            message = token.choices[0].delta.content
            yield message


def get_response_and_dataframe(messages, stream):
    for message in get_response(messages, stream):
        messages[-1]['content'] += message
        yield convert_message_to_dataframe(messages[:-1])
        yield convert_message_to_dataframe(messages)


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
    return []


def convert_message_to_dataframe(messages):
    messages_ = []
    for message in messages:  # After edit, the \n will miss
        messages_.append({"role": message["role"], "content": message["content"].replace("\n", "\n\u0000")})
    return pd.DataFrame.from_records(messages_)


def change_dataframe_to_current_message(dataframe):
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
    return convert_message_to_dataframe(messages=current_messages)


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
    start_button.click(fn=chatbot_interface, outputs=[outputs])
    save_button.click(fn=save_func)
    del_button.click(fn=delete_func, outputs=[outputs])
    continue_button.click(fn=change_dataframe_to_current_message, inputs=[outputs], outputs=[outputs])

    # commit_btn.click(fn=add_text, inputs=[chatbot, txt], outputs=[chatbot, txt]).then(
    #     bot, chatbot, chatbot, api_name="bot_response"
    # ).then(lambda: gr.Textbox(interactive=True), None, [txt], queue=False)

demo.launch(server_name="0.0.0.0",
            share=False,
            debug=True,
            )

# https://blog.csdn.net/jclian91/article/details/132417892
