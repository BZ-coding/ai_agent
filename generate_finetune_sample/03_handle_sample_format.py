import json
import os
import uuid

from tqdm import tqdm

SAMPLES_PATH = 'checked_finetune_sample.txt'
FORMATED_SAMPLES_PATH = 'checked_finetune_sample_formatted.jsonl'

dir_path = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(dir_path, SAMPLES_PATH), 'r') as f:
    queries = f.readlines()

# 将数据保存为JSONL文件
with open(os.path.join(dir_path, FORMATED_SAMPLES_PATH), 'w') as f:
    for query in tqdm(queries):
        query = query.strip()
        data = {
            "id": str(uuid.uuid4()),
            "query": query,
            "messages": [],
            "is_verify": False,
        }
        f.write(json.dumps(data, ensure_ascii=False) + '\n')
