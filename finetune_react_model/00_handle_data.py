import json
import os

FORMATED_SAMPLES_PATH = 'generate_finetune_sample/checked_finetune_sample_formatted.jsonl'
OUTPUT_FINETUNE_SAMPLES_PATH = 'finetune_react_model/finetune_samples.jsonl'
OUTPUT_FINETUNE_TEST_SAMPLES_PATH = 'finetune_react_model/finetune_test_samples.jsonl'

with (open(FORMATED_SAMPLES_PATH, 'r') as f, open(OUTPUT_FINETUNE_SAMPLES_PATH, 'w') as f_out,
      open(OUTPUT_FINETUNE_TEST_SAMPLES_PATH, 'w') as f_test_out):
    for i, line in enumerate(f):
        data = json.loads(line)
        if data["is_verify"]:
            out_data = {"messages": data["messages"]}
            data_len = len(str(out_data))
            if data_len > 9000:  # 样本长度太长，哪怕用lora+offload，也会oom
                print(f"index:{i} len:{data_len} to f_test_out.")
                f_test_out.write(json.dumps(out_data, ensure_ascii=False) + '\n')
            else:
                print(f"index:{i} len:{data_len} to f_out.")
                f_out.write(json.dumps(out_data, ensure_ascii=False) + '\n')
