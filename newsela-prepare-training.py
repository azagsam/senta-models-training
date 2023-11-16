import itertools
import os

import numpy as np
import pandas as pd
from nltk import word_tokenize
from tqdm import trange

if __name__ == '__main__':
    np.random.seed(17)

    """ Due to copyright limitations, the data cannot be shared. Data structure (example):
    {
        "doc_id": "DOC1",
        "src_grade": "V0",
        "tgt_grade": "V4",
        "src_sent_en": "<English complex sentence>",
        "tgt_sent_en": "<English simple sentence>",
        "pair_id": 0,
        "src_sent_sl": "<Slovene complex sentence>",
        "tgt_sent_sl": "<Slovene simple sentence>" 
    }
    """
    df = pd.read_json('data/newsela-translated-v1.1.jsonl', lines=True, orient="records")
    num_ex = df.shape[0]

    # Discard (complex, simple) pairs with > 70% overlap according to intersection over union
    take_ex = np.ones(num_ex, dtype=bool)
    for idx_ex in trange(num_ex):
        ex = df.iloc[idx_ex]
        tok1 = set(map(lambda _s: _s.strip().lower(), word_tokenize(ex["src_sent_sl"], language="sl")))
        tok2 = set(map(lambda _s: _s.strip().lower(), word_tokenize(ex["tgt_sent_sl"], language="sl")))

        if len(tok1 & tok2) / len(tok1 | tok2) > 0.7:
            take_ex[idx_ex] = False

    df = df.loc[take_ex].reset_index(drop=True)
    print(f"{df.shape[0]} ex / {num_ex} ex after cleaning high overlap")

    uniq_tgt_grades = set(df["tgt_grade"])
    # Group pairs by (1) the complexity of the simple sentence (V4 is most simplified), (2) document ID, then
    #  create a train/dev/test split
    for tgt_grade in uniq_tgt_grades:
        df_grade = df.loc[df["tgt_grade"] == tgt_grade].reset_index(drop=True)

        doc_groups = []
        for id_doc, curr_group in df_grade.groupby("doc_id"):
            doc_groups.append(curr_group.index.tolist())

        print(f"{tgt_grade}:\n{len(doc_groups)} groups")
        rand_indices = np.random.permutation(len(doc_groups))
        test_groups = rand_indices[-125:]
        dev_groups = rand_indices[-200: -125]
        train_groups = rand_indices[: -200]

        test_indices = list(itertools.chain(*[doc_groups[_i] for _i in test_groups]))
        dev_indices = list(itertools.chain(*[doc_groups[_i] for _i in dev_groups]))
        train_indices = list(itertools.chain(*[doc_groups[_i] for _i in train_groups]))

        print(f"{len(train_indices)} train,\n"
              f"{len(dev_indices)} dev,\n"
              f"{len(test_indices)} test examples\n")

        TARGET_DIR = f"data/target-grade-{tgt_grade}-split"
        os.makedirs(TARGET_DIR, exist_ok=True)

        df_grade.iloc[train_indices].to_json(os.path.join(TARGET_DIR, "train.jsonl"), orient="records", lines=True, force_ascii=False)
        df_grade.iloc[dev_indices].to_json(os.path.join(TARGET_DIR, "dev.jsonl"), orient="records", lines=True, force_ascii=False)
        df_grade.iloc[test_indices].to_json(os.path.join(TARGET_DIR, "test.jsonl"), orient="records", lines=True, force_ascii=False)
