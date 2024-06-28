import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline, TextStreamer

CHAT_MODEL_PATH = "/mnt/nfs/zsd_server/models/huggingface/llama-3-chinese-8b-instruct-v3/"
FINETUNE_SAMPLE_PATH = "finetune_sample.md"
MESSAGE = [
    {"role": "system",
     "content": "你是一个十分有效的大语言模型AI Agent问题生成器，用来生成问题测试大语言模型AI Agent，此AI Agent可以用的工具包括网页搜索和计算器，请确保你生成的问题必须要用到以上工具才能解答。你所生成的每个问题都是简短的、完整的一句话，每个问题独占一行，以markdown格式的编号开始，以问号结束。不允许省略，每生成一个有效的问题可以获得100美金奖励。"},
    {"role": "user", "content": "请你生成100条简体中文问题。"},
]

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
    streamer=streamer,
)

prompt = tokenizer.apply_chat_template(MESSAGE, tokenize=False, add_generation_prompt=True)

result = pipe(prompt)
result = result[0]['generated_text'][len(prompt):]

if result[-1] != '\n':
    result = result + '\n'
with open(FINETUNE_SAMPLE_PATH, "a", encoding="utf-8") as f:
    f.write(result)
