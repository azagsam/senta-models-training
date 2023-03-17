import pandas as pd

org = pd.read_json('output/rtvslo-original.jsonl', lines=True)
eno = pd.read_json('output/rtvslo-enostavno.jsonl', lines=True)

df = pd.merge(org, eno, how='outer', on='body')
df = df[['body']]
df.to_json('output/doc2vec-data.jsonl', lines=True, force_ascii=False, orient='records')