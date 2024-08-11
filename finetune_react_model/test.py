# https://huggingface.co/docs/trl/lora_tuning_peft
# https://huggingface.co/docs/trl/sft_trainer
# https://github.com/huggingface/trl/blob/main/examples/research_projects/stack_llama/scripts/supervised_finetuning.py
# https://blog.csdn.net/chaishen10000/article/details/133150584
# trl how to train conversational format ?
from transformers import AutoTokenizer

model_path = "/mnt/nfs/zsd_server/models/huggingface/llama-3-chinese-8b-instruct-v3"

tokenizer = AutoTokenizer.from_pretrained(model_path)

data = [
    {"role": "system", "content": "123"},
    {"role": "user", "content": "123"},
    {"role": "assistant", "content": "123"},
    {"role": "tools", "content": "123"},
]
result = tokenizer.apply_chat_template(conversation=data, tokenize=False, add_generation_prompt=False, return_tensors="pt")
print(result)
