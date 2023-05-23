import os

import pandas as pd
import numpy as np


def full_dataset():
    train, val, test = np.split(df.sample(frac=1, random_state=42), [int(.98 * len(df)), int(.99 * len(df))])
    train.to_json('data/newsela_data/newsela-translated/train.jsonl', lines=True, force_ascii=False, orient='records')
    val.to_json('data/newsela_data/newsela-translated/val.jsonl', lines=True, force_ascii=False, orient='records')
    test.to_json('data/newsela_data/newsela-translated/test.jsonl', lines=True, force_ascii=False, orient='records')


def no_src_duplicates():
    df_dedup = df.drop_duplicates(subset="src_sent_sl")
    train, val, test = np.split(df_dedup.sample(frac=1, random_state=42), [int(.96 * len(df_dedup)), int(.98 * len(df_dedup))])
    train.to_json('data/newsela_data/newsela-translated/src_dedup/train.jsonl', lines=True, force_ascii=False, orient='records')
    val.to_json('data/newsela_data/newsela-translated/src_dedup/val.jsonl', lines=True, force_ascii=False, orient='records')
    test.to_json('data/newsela_data/newsela-translated/src_dedup/test.jsonl', lines=True, force_ascii=False, orient='records')


def grades():
    tgt_grades = set(df['tgt_grade'].unique())
    df['doc_id_int'] = df['doc_id'].apply(lambda x: int(x[3:]))
    for grade in tgt_grades:
        graded_df = df[df['tgt_grade'] == grade]
        train, val, test = graded_df[graded_df['doc_id_int'] < 1000], graded_df[(graded_df['doc_id_int'] >= 1000) & (graded_df['doc_id_int'] <= 1020)], graded_df[graded_df['doc_id_int'] > 1020]
        print(len(train), len(val), len(test))
        os.makedirs(f'data/newsela_data/newsela-translated/target-grade-{grade}-dedup', exist_ok=True)
        train.to_json(f'data/newsela_data/newsela-translated/target-grade-{grade}-dedup/train.jsonl', lines=True, force_ascii=False,
                      orient='records')
        val.to_json(f'data/newsela_data/newsela-translated/target-grade-{grade}-dedup/val.jsonl', lines=True, force_ascii=False,
                    orient='records')
        test.to_json(f'data/newsela_data/newsela-translated/target-grade-{grade}-dedup/test.jsonl', lines=True, force_ascii=False,
                     orient='records')


def one2many():
    df['doc_id_int'] = df['doc_id'].apply(lambda x: int(x[3:]))
    subset = df[df['src_grade'] == 'V0']
    subset['src_sent_sl'] = subset['src_sent_sl'] + ' ' + subset['tgt_grade'].apply(lambda x: '[' + x + ']')
    train, val, test = subset[subset['doc_id_int'] < 1000], subset[(subset['doc_id_int'] >= 1000) & (subset['doc_id_int'] <= 1020)], subset[subset['doc_id_int'] > 1020]
    print(len(train), len(val), len(test))
    os.makedirs(f'data/newsela_data/newsela-translated/target-grade-one2many', exist_ok=True)
    train.to_json(f'data/newsela_data/newsela-translated/target-grade-one2many/train.jsonl', lines=True,
                  force_ascii=False,
                  orient='records')
    val.to_json(f'data/newsela_data/newsela-translated/target-grade-one2many/val.jsonl', lines=True,
                force_ascii=False,
                orient='records')
    test.to_json(f'data/newsela_data/newsela-translated/target-grade-one2many/test.jsonl', lines=True,
                 force_ascii=False,
                 orient='records')


def chatgpt_prompts():
    graded_df = df[(df['src_grade'] == 'V0') & (df['tgt_grade'] == 'V4')]
    graded_df = graded_df.sample(frac=1)
    print('Poenostavi stavke glede na zglede:\n\n')
    for idx, row in graded_df.sample(n=10).iterrows():
        print(idx, '"', row['src_sent_sl'], '"',  '->', '"', row['tgt_sent_sl'], '"')
    for idx, row in graded_df.sample(n=1).iterrows():
        print('\n', '"', row['src_sent_sl'], '"',  '->')

    print()



if __name__ == '__main__':
    df = pd.read_json('data/newsela_data/newsela-translated/translations.jsonl', lines=True)
    df['doc_id'][0] = 'DOC1'
    # full_dataset()
    # no_src_duplicates()
    # grades()
    # chatgpt_prompts()
    one2many()
