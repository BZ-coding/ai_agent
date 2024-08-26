import spaces
import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
import gradio as gr

from utils.chatbot_local import ChatBot

MODEL_PATH = 'lora_adapter'

model = AutoPeftModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

chatbot = ChatBot(model=model, tokenizer=tokenizer)

PLACEHOLDER = """
<center>
<p>Hi! How can I help you today?</p>
</center>
"""

CSS = """
.duplicate-button {
    margin: auto !important;
    color: white !important;
    background: black !important;
    border-radius: 100vh !important;
}
h3 {
    text-align: center;
}
"""


@spaces.GPU()
def stream_chat(
        message: str,
        history: list,
):
    print(f'message: {message}')
    print(f'history: {history}')

    conversation = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
    for prompt, answer in history:
        conversation.extend([
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ])

    conversation.append({"role": "user", "content": message})

    buffer = ""
    for token in chatbot.chat(messages=conversation, stream=True):
        buffer += token
        yield buffer


gr_chatbot = gr.Chatbot(height=600, placeholder=PLACEHOLDER)

with gr.Blocks(css=CSS, theme="soft") as demo:
    gr.ChatInterface(
        fn=stream_chat,
        chatbot=gr_chatbot,
        fill_height=True,
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        share=False,
        debug=True,
    )
