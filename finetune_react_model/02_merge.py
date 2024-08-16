from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel

model_path = "/mnt/nfs/zsd_server/models/huggingface/llama-3-chinese-8b-instruct-v3"
lora_path = "/mnt/nfs/zsd_server/models/my/llama-3-chinese-8b-tools"
output_path = "/mnt/nfs/zsd_server/models/my/llama-3-chinese-8b-tools/merged"

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
)
tokenizer = AutoTokenizer.from_pretrained(model_path)
lora_model = PeftModel.from_pretrained(
    model,
    lora_path,
    torch_dtype=torch.float16,
)

model = lora_model.merge_and_unload()
model.save_pretrained(output_path)
tokenizer.save_pretrained(output_path)

# cd llama.cpp
# python convert_hf_to_gguf.py /mnt/nfs/zsd_server/models/my/llama-3-chinese-8b-tools/merged
## FROM /mnt/nfs/zsd_server/models/my/llama-3-chinese-8b-tools/merged/Merged-8.0B-F16.gguf
# ollama create llama-3-chinese-8B-tools-F16-LoRA -f ModelFile.txt
