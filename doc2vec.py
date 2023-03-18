import pandas as pd
from gensim.models.doc2vec import Doc2Vec
from sklearn.metrics.pairwise import cosine_similarity


def d2v_model(model, row, candidates):
    # calculate embeddings
    source_vector = model.infer_vector(row['body'].split())
    target_vectors = []
    for idx_can, row_can in candidates.iterrows():
        vec = model.infer_vector(row_can['body'].split())
        target_vectors.append(vec)

    # compute cosine similarity
    similarity_matrix = cosine_similarity(source_vector.reshape(1, -1), target_vectors)
    similarity_matrix_reshaped = similarity_matrix[0]
    candidates['similarity'] = similarity_matrix_reshaped
    candidates.sort_values(by=['similarity'], ascending=False, inplace=True)

    return candidates

