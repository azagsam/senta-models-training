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
    for grade in tgt_grades:
        graded_df = df[df['tgt_grade'] == grade]
        train, val, test = np.split(graded_df.sample(frac=1, random_state=42),
                                    [int(.9 * len(graded_df)), int(.95 * len(graded_df))])
        print(len(train), len(val), len(test))
        os.makedirs(f'data/newsela_data/newsela-translated/target-grade-{grade}', exist_ok=True)
        train.to_json(f'data/newsela_data/newsela-translated/target-grade-{grade}/train.jsonl', lines=True, force_ascii=False,
                      orient='records')
        val.to_json(f'data/newsela_data/newsela-translated/target-grade-{grade}/val.jsonl', lines=True, force_ascii=False,
                    orient='records')
        test.to_json(f'data/newsela_data/newsela-translated/target-grade-{grade}/test.jsonl', lines=True, force_ascii=False,
                     orient='records')


if __name__ == '__main__':
    df = pd.read_json('data/newsela_data/newsela-translated/translations.jsonl', lines=True)
    # full_dataset()
    # no_src_duplicates()
    grades()
