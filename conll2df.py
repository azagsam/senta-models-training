import conllu
import pandas as pd
import os


docs = []
for root, dirs, files in os.walk("data/rtvslo"):
    path = root.split(os.sep)
    print((len(path) - 1) * '---', os.path.basename(root))
    for file in files:
        print(file)
        full_path = os.path.join(root, file)
        with open(full_path, encoding='utf-8') as f:
            data = f.read()
            doc = conllu.parse(data)
            article_id, article_metadata, article_body = '', {}, []
            for sent in doc:
                if 'newdoc id' in sent.metadata:
                    if not docs and not article_id:  # first document
                        article_id = sent.metadata['newdoc id']
                        article_metadata = sent.metadata
                    else:
                        if article_id != sent.metadata['newdoc id']:  # save previous document
                            body = " ".join(article_body)
                            article_metadata['body'] = body
                            docs.append(article_metadata)

                            # start new document
                            article_id = sent.metadata['newdoc id']
                            article_body = []
                            article_metadata = sent.metadata

                elif 'sent_id' in sent.metadata:
                    article_body.append(sent.metadata['text'])

df = pd.DataFrame(docs)
df.drop(columns=['sent_id', 'newpar id', 'text'], inplace=True)
df.to_json('output/rtvslo-original.jsonl', lines=True, force_ascii=False, orient='records')