import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def headline_model(model, row, candidates):
    # calculate embeddings
    source_vector = model.encode(row['title'])
    target_vectors = []
    for idx_can, row_can in tqdm(candidates.iterrows()):
        if not row_can['title']:  # empty body
            target_vectors.append(np.random.rand(768))
            continue
        target_vector = model.encode(row_can['title'])
        target_vectors.append(target_vector)

    # compute cosine similarity
    similarity_matrix = cosine_similarity(source_vector.reshape(1, -1), target_vectors)
    similarity_matrix_reshaped = similarity_matrix[0]
    candidates['similarity'] = similarity_matrix_reshaped
    candidates.sort_values(by=['similarity'], ascending=False, inplace=True)

    return candidates