import pandas as pd
from gensim.models.doc2vec import Doc2Vec
from sklearn.metrics.pairwise import cosine_similarity


# get data
org = pd.read_json('output/rtvslo-original.jsonl', lines=True)
eno = pd.read_json('output/rtvslo-enostavno.jsonl', lines=True)
org['date'] = pd.to_datetime(org['date'])
eno.dropna(subset=['date'], inplace=True)  # drop na values from
eno['date'] = pd.to_datetime(eno['date'])

# load model
fname = "model/doc2vec/model"
model = Doc2Vec.load(fname)

for idx, row in eno.iterrows():
    # filter data
    end_date = row['date']
    start_date = end_date - pd.DateOffset(days=5)
    window = pd.date_range(start=start_date, end=end_date)
    candidates = org[org['date'].isin(window)].reset_index()

    # calculate embeddings
    source_vector = model.infer_vector(row['body'].split())
    target_vectors = []
    for idx_can, row_can in candidates.iterrows():
        vec = model.infer_vector(row_can['body'].split())
        target_vectors.append(vec)

    # computer cosine similarity
    similarity_matrix = cosine_similarity(source_vector.reshape(1, -1), target_vectors)
    similarity_matrix_reshaped = similarity_matrix[0]
    candidates['similarity'] = similarity_matrix_reshaped

    candidates.sort_values(by=['similarity'], ascending=False, inplace=True)
    print('------------------------')
    print('ENOSTAVNO:', row['title'], '\n')
    for title in candidates['title'][:5]:
        print(title, '\n\n')

