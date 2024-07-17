import random
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline, TextStreamer

CHAT_MODEL_PATH = "/mnt/nfs/zsd_server/models/huggingface/llama-3-chinese-8b-instruct-v3/"
FINETUNE_SAMPLE_PATH = "finetune_sample.md"
CHECKED_SAMPLE_PATH = "checked_finetune_sample.txt"
MESSAGE = [
    {"role": "system",
     "content": "你是一个十分有效的大语言模型AI Agent问题判别器，用来判别给定的问题能否用来测试AI Agent。你回答的第一个字只能是\"是\"或者\"否\"，从第二句开始，你可以解释自己的理由。请注意，此AI Agent可以用的工具包括网页搜索和计算器，所以只有当你认为此问题必须要用到以上工具才能解答时，这个问题才算是一个合格的问题，此时你应该回答是。除此之外别的情况都请回答否，特别是该问题可以不用工具而被直接回答时或者该问题并不是一个真的问题时，你也应该回答否。"},
]
RANDOM_PRINT = True

chat_model = AutoModelForCausalLM.from_pretrained(
    CHAT_MODEL_PATH,
    # load_in_8bit=True,
    device_map='auto',
    torch_dtype=torch.float16,  # 推理时用fp16精度更高，训练时要用bf16不容易精度溢出
)
tokenizer = AutoTokenizer.from_pretrained(CHAT_MODEL_PATH)

streamer = TextStreamer(tokenizer)
pipe = pipeline(
    "text-generation",
    model=chat_model,
    tokenizer=tokenizer,
    max_length=4096,
    truncation=True,
    repetition_penalty=1.2,
    do_sample=True,
    temperature=0.6,
    top_p=0.95,
    # streamer=streamer,
)

with open(FINETUNE_SAMPLE_PATH, "r", encoding="utf-8") as f:
    samples = f.readlines()
with open(CHECKED_SAMPLE_PATH, "w", encoding="utf-8") as f:
    for sample in samples:
        sample = sample.strip()
        if not sample:
            continue
        result = re.match(r'\d+\.\s*(.+)', sample)
        if not result:
            continue
        sample = result.group(1)

        message = MESSAGE.copy()
        message.append({"role": "user", "content": f"请判别\"{sample}\"是一个有效的问题吗？"})

        prompt = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        result = pipe(prompt)
        result = result[0]['generated_text'][len(prompt):]

        if result[0] != "是":
            print(result, f"问题: {sample}", "\n", sep="\n")
            continue
        if RANDOM_PRINT and random.randint(0, 10) == 1:
            print(result, f"问题: {sample}", "\n", sep="\n")

        f.write(sample + '\n')
