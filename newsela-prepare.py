import pandas as pd
from tqdm import tqdm
import json
import deepl

# import and drop na
df = pd.read_csv('data/newsela_data/newsela_data_share-20150302/newsela_data_share-20150302/newsela_articles_20150302.aligned.sents.txt',
                 sep='\t',
                 on_bad_lines='warn',
                 encoding='utf-8',
                 header=None)
df = df.dropna()
df.columns = ['doc_id', 'src_grade', 'tgt_grade', 'src_sent_en', 'tgt_sent_en']
df['pair_id'] = list(range(len(df)))

df.to_json('data/newsela_data/newsela-aligned-sents.jsonl', lines=True, force_ascii=False, orient='records')

# num of chars
print(df['src_sent_en'].apply(len).sum() + df['tgt_sent_en'].apply(len).sum())


