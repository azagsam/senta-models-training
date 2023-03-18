import pandas as pd
from gensim.models import Doc2Vec
from sentence_transformers import SentenceTransformer
from doc2vec import d2v_model
from sent_emb import body_model
from headlines import headline_model
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def main():
    # get data
    org = pd.read_json('output/rtvslo-original.jsonl', lines=True)
    eno = pd.read_json('output/rtvslo-enostavno.jsonl', lines=True)
    org['date'] = pd.to_datetime(org['date'])
    eno.dropna(subset=['date'], inplace=True)  # drop na values from
    eno['date'] = pd.to_datetime(eno['date'])

    # load model
    sent_pretrained_model = SentenceTransformer('sentence-transformers/LaBSE')
    d2v_pretrained_model = Doc2Vec.load('model/doc2vec/model')

    with open('results-sentence.txt', mode='w', encoding='utf-8') as f:
        eno = eno.sample(frac=1).reset_index(drop=True)
        for idx, row in eno.iterrows():
            # filter data
            end_date = row['date']
            start_date = end_date - pd.DateOffset(days=5)
            window = pd.date_range(start=start_date, end=end_date)
            candidates = org[org['date'].isin(window)].reset_index()

            headline_results = headline_model(sent_pretrained_model, row, candidates)
            body_unranked_results = body_model(sent_pretrained_model, row, candidates, ranking=False, n_sent=0)
            body_ranked_results = body_model(sent_pretrained_model, row, candidates, ranking=True, n_sent=10)
            d2v_results = d2v_model(d2v_pretrained_model, row, candidates)

            print()

            results = {'headline': headline_results,
                       'body_unranked': body_unranked_results,
                       'body_ranked': body_ranked_results,
                       'doc2vec': d2v_results
                       }
            for exp, result in results.items():
                print('------------------------')
                print(exp.upper(), '\n\n')
                print('ENOSTAVNO:', row['title'], '\n\n')
                for title in result['title'][:5]:
                    print(title, '\n\n')



if __name__ == '__main__':
    main()