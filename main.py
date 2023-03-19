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

    eno = eno.sample(frac=1, random_state=42).reset_index(drop=True)
    days = 7
    num_of_candidates = 10
    print(f'Time window: {days} days ')
    for idx, row in eno.iterrows():
        # filter data
        end_date = row['date']
        start_date = end_date - pd.DateOffset(days=days)
        window = pd.date_range(start=start_date, end=end_date)
        candidates = org[org['date'].isin(window)].reset_index()

        # calculate
        headline_results = headline_model(sent_pretrained_model, row, candidates.copy(deep=True))
        body_unranked_results = body_model(sent_pretrained_model, row, candidates.copy(deep=True), ranking=False, n_sent=0)
        body_ranked_results = body_model(sent_pretrained_model, row, candidates.copy(deep=True), ranking=True, n_sent=10)
        d2v_results = d2v_model(d2v_pretrained_model, row, candidates.copy(deep=True))

        # results
        results = {'headline': headline_results,
                   'body_unranked': body_unranked_results,
                   'body_ranked': body_ranked_results,
                   'doc2vec': d2v_results
                   }
        for exp, result in results.items():
            print('------------------------')
            print(exp.upper(), '\n\n')
            print('ENOSTAVNO:', row['date'], row['title'], '\n\n')
            for can_idx, can_row in result[:num_of_candidates].iterrows():
                print(can_row['date'], can_row['title'], can_row['similarity'], '\n')


if __name__ == '__main__':
    main()