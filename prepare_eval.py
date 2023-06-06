import os

import pandas as pd
from collections import defaultdict
import json


def combine():
    d = defaultdict(list)
    for root, dirs, files in os.walk("data/eval"):
        for file in files:
            file_path = os.path.join(root, file)
            if '.txt' in file:
                with open(file_path, 'r') as f:
                    text = f.read()
                    d['text'].append(text)
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = f.read()
                    data = json.loads(data)
                    d['text'].append(data['body'])
            d['path'].append(file_path)
            d['source'].append(os.path.split(root)[-1])

    df = pd.DataFrame(d)
    df['id'] = list(range(len(df)))
    df.to_json('data/eval/eval_datasets.jsonl', lines=True, force_ascii=False, orient='records')


def eval_small():
    df = pd.read_json('data/eval/eval_datasets.jsonl', lines=True)
    sample = df.sample(n=10)
    sample.to_json('data/eval/eval_small.jsonl', lines=True, force_ascii=False, orient='records')


if __name__ == '__main__':
    # combine()
    eval_small()