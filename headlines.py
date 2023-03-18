import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


# get data
org = pd.read_json('output/rtvslo-original.jsonl', lines=True)
eno = pd.read_json('output/rtvslo-enostavno.jsonl', lines=True)
org['date'] = pd.to_datetime(org['date'])
eno.dropna(subset=['date'], inplace=True)  # drop na values from
eno['date'] = pd.to_datetime(eno['date'])

# load model
model = SentenceTransformer('sentence-transformers/LaBSE')

with open('results-sentence.txt', mode='w', encoding='utf-8') as f:

    eno = eno.sample(frac=1).reset_index(drop=True)
    for idx, row in eno.iterrows():
        # filter data
        end_date = row['date']
        start_date = end_date - pd.DateOffset(days=5)
        window = pd.date_range(start=start_date, end=end_date)
        candidates = org[org['date'].isin(window)].reset_index()

        # calculate embeddings
        source_vector = model.encode(row['title'])
        target_vectors = []
        for idx_can, row_can in tqdm(candidates.iterrows()):
            if not row_can['title']:  # empty body
                target_vectors.append(np.random.rand(768))
                continue
            target_vector = model.encode(row_can['title'])
            target_vectors.append(target_vector)

        # computer cosine similarity
        similarity_matrix = cosine_similarity(source_vector.reshape(1, -1), target_vectors)
        similarity_matrix_reshaped = similarity_matrix[0]
        candidates['similarity'] = similarity_matrix_reshaped

        candidates.sort_values(by=['similarity'], ascending=False, inplace=True)
        print('------------------------')
        print(f'ENOSTAVNO: {row["title"]} \n')
        f.write(f'ENOSTAVNO: {row["title"]} \n')
        for idx_can, title in enumerate(candidates['title'][:5]):
            message = f'{idx_can}: {title} \n'
            print(message)
            f.write(message)