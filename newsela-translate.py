from tqdm import tqdm
import json
import deepl
import pandas as pd

df = pd.read_json('data/newsela_data/newsela-aligned-sents.jsonl', lines=True)
auth_key = "d87de57b-e9a8-9997-5bdb-39efc7a7934d"
translator = deepl.Translator(auth_key)

cached_translations = {}
for idx, row in tqdm(df.iterrows()):
    row = row.to_dict()

    # translate source
    if row['src_sent_en'] in cached_translations.keys():
        row['src_sent_sl'] = cached_translations[row['src_sent_en']]
    else:
        result_source = translator.translate_text(row['src_sent_en'], source_lang="EN", target_lang="SL")
        row['src_sent_sl'] = result_source.text
        cached_translations[row['src_sent_en']] = result_source.text

    # translate target
    if row['tgt_sent_en'] in cached_translations.keys():
        row['tgt_sent_sl'] = cached_translations[row['tgt_sent_en']]
    else:
        result_target = translator.translate_text(row['tgt_sent_en'], source_lang="EN", target_lang="SL")
        row['tgt_sent_sl'] = result_target.text
        cached_translations[row['tgt_sent_en']] = result_target.text

    with open('translations.jsonl', 'a', encoding='utf8') as f:
        json.dump(row, f, ensure_ascii=False)
        f.write('\n')
