import os
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
import numpy as np

import classla
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def splt():
    df = pd.read_json('data/mverb.jsonl', lines=True)
    train, val, test = np.split(df.sample(frac=1, random_state=42), [int(.98 * len(df)), int(.99 * len(df))])
    train.to_json('data/cckres-mverb/train.jsonl', lines=True, force_ascii=False, orient='records')
    val.to_json('data/cckres-mverb/val.jsonl', lines=True, force_ascii=False, orient='records')
    test.to_json('data/cckres-mverb/test.jsonl', lines=True, force_ascii=False, orient='records')


def main():
    d = defaultdict(list)
    for file in tqdm(os.listdir('data/cckresV1_0-text')):
        with open(os.path.join('data/cckresV1_0-text', file)) as f:
            for line in f:
                target = []
                doc = nlp(line)
                for sent in doc.sentences:
                    if len(sent.words) < 15:
                        contains_verb = False
                        for word in sent.words:
                            if word.upos == 'VERB' or word.upos == 'AUX':
                                contains_verb = True
                                continue
                            else:
                                target.append(word.text)
                        # add as an example
                        if contains_verb:
                            d['src_sent'].append(" ".join(target))
                            d['tgt_sent'].append(sent.text)
    df = pd.DataFrame(d)
    df.to_json('data/mverb.jsonl', lines=True, force_ascii=False, orient='records')


if __name__ == '__main__':
    nlp = classla.Pipeline('sl', processors='tokenize,pos')  # initialize the default Slovenian pipeline, use hr for Croatian, sr for Serbian, bg for Bulgarian, mk for Macedonian

    # main()
    splt()

