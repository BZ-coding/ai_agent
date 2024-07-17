import gradio as gr
from openai import OpenAI

if gr.NO_RELOAD:
    client = OpenAI(
        base_url='http://localhost:11434/v1/',
        api_key='ollama',  # required but ignored
    )


def get_response(**params):
    chat_completion = client.chat.completions.create(
        **params,
        model='llama-3-chinese-8b-instruct-v3-f16',
    )
    return chat_completion


def run_conversation(query: str, stream=False, max_retry=5):
    params = {
        "messages": [{"role": "user", "content": query}],
        "temperature": 0.6,
        "stream": stream,
    }
    response = get_response(**params)
    return response


def chatbot_interface(query):
    stream = True
    response = run_conversation(query, stream=stream)
    if not stream:
        return response.choices[0].message.content
    message = ""
    for token in response:
        if token.choices[0].finish_reason is not None:
            continue
        message += token.choices[0].delta.content
        yield message


with gr.Blocks() as demo:
    gr.Markdown("# Manually Annotate Samples")
    with gr.Row():
        inputs = gr.Textbox(placeholder="Enter your query here..."),
        response = gr.Textbox(label="Response")
    submit_button = gr.Button("提交")
    submit_button.click(fn=chatbot_interface, inputs=inputs[0], outputs=response)

demo.launch(server_name="0.0.0.0",
            share=True,
            # debug=True,
            )
