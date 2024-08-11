import json
import os

FORMATED_SAMPLES_PATH = 'generate_finetune_sample/checked_finetune_sample_formatted.jsonl'
OUTPUT_FINETUNE_SAMPLES_PATH = 'finetune_react_model/finetune_samples.jsonl'

with open(FORMATED_SAMPLES_PATH, 'r') as f, open(OUTPUT_FINETUNE_SAMPLES_PATH, 'w') as f_out:
    for line in f:
        data = json.loads(line)
        if data["is_verify"]:
            out_data = {"messages": data["messages"]}
            f_out.write(json.dumps(out_data, ensure_ascii=False) + '\n')
