import os
import sys

import torch
import transformers
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import wandb

# Add requirement for wandb core
wandb.require("core")
wandb.init(
    project="finetune_react_model",  # https://wandb.ai/bz-zhangshengdong/finetune_react_model/workspace
    name="数据量160",
    group="加入提炼事实的数据",
    magic=True,
)
epoch = 3.0

print(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from finetune_react_model.utils.dataset import handle_multi_conversation_dataset

MODEL_PATH = "/mnt/nfs/zsd_server/models/huggingface/llama-3-chinese-8b-instruct-v3"
SAVE_PATH = "/mnt/nfs/zsd_server/models/my/llama-3-chinese-8b-tools"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
# print(tokenizer)

dataset = load_dataset("json", data_files="./finetune_react_model/finetune_samples.jsonl")
dataset = handle_multi_conversation_dataset(dataset=dataset, tokenizer=tokenizer)
print(dataset)
dataset_eval = load_dataset("json", data_files="./finetune_react_model/finetune_test_samples.jsonl")
dataset_eval = handle_multi_conversation_dataset(dataset=dataset_eval, tokenizer=tokenizer)
print(dataset_eval)

data_collator = transformers.DataCollatorForSeq2Seq(
    tokenizer,
    return_tensors="pt",
    padding=True,
    pad_to_multiple_of=1,
    # pad_to_multiple_of=ARGS.max_length,  # the max_length arg is unused to padding label
)

if os.path.exists(SAVE_PATH):
    os.system(f"rm -rf {SAVE_PATH}")  # 在nfs上shutil.rmtree会报正忙、非空
os.makedirs(SAVE_PATH, exist_ok=True)

training_arguments = transformers.TrainingArguments(
    save_total_limit=1,
    load_best_model_at_end=False,
    auto_find_batch_size=False,
    do_train=True,
    overwrite_output_dir=True,
    save_safetensors=True,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=epoch,
    learning_rate=1.0e-5,  # qwen是全参7e-6，我用了lora，所以学习率要大一点
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    output_dir=SAVE_PATH,
    bf16=True,
    logging_steps=1,
    report_to="wandb",  # https://docs.wandb.ai/guides/integrations/huggingface
    # fsdp="full_shard offload",
    # deepspeed="/mnt/nfs/zsd_server/codes/ai_agent/finetune_react_model/ds_config.json",
    eval_strategy="epoch",
    per_device_eval_batch_size=1,
)
print(training_arguments)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    # load_in_8bit=True,
    # torch_dtype=torch.float16,
    torch_dtype=torch.bfloat16,
    # device_map="cuda",  # "auto"
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.enable_input_require_grads()
model.print_trainable_parameters()

model.config.use_cache = False
model.gradient_checkpointing_enable()

print(model)

trainer = transformers.Trainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset_eval["train"],
    args=training_arguments,
    data_collator=data_collator,
)

trainer.train()

model.save_pretrained(SAVE_PATH, safe_serialization=True)
tokenizer.save_pretrained(SAVE_PATH)

# ollama create llama-3-chinese-8B-tools-F16-LoRA -f /mnt/nfs/zsd_server/codes/ai_agent/finetune_react_model/ModelFile.txt
