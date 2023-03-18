import networkx as nx
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
from nltk.tokenize import sent_tokenize
import nltk
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def rank(model, text: str, n: int):
    # tokenize
    sentences = nltk.sent_tokenize(text, language='slovene')

    # Compute the sentence embeddings
    embeddings = model.encode(sentences, convert_to_numpy=True, batch_size=128)

    # similarity matrix
    sim_mat = cosine_similarity(embeddings)

    # rescale
    scaler = MinMaxScaler(feature_range=(0, 1))
    sim_mat = scaler.fit_transform(sim_mat.flatten().reshape(-1, 1)).reshape(len(embeddings), len(embeddings))

    # calculate pagerank
    nx_graph = nx.from_numpy_array(sim_mat)
    scores = nx.pagerank(nx_graph, alpha=0.85, max_iter=500)  # number of cycles to converge
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)

    # reorder sentences
    idx_sentences = []
    for _, sentence in ranked_sentences[:n]:
        idx = sentences.index(sentence)
        idx_sentences.append((idx, sentence))
    idx_sentences.sort()

    return [sent for _, sent in idx_sentences]


def average_vectors(vectors):
    vectors_np = np.array(vectors)
    avg_vector = np.mean(vectors_np, axis=0)
    return avg_vector


def body_model(model, row, candidates, ranking, n_sent):
    # ranking
    if ranking:
        source_sentences = rank(model, row['body'], n=n_sent)
    else:
        source_sentences = sent_tokenize(row['body'], language='slovene')

    source_vectors = model.encode(source_sentences)
    source_vector = average_vectors(source_vectors)
    target_vectors = []
    for idx_can, row_can in tqdm(candidates.iterrows()):

        if not row_can['body']:  # empty body
            target_vectors.append(np.random.rand(768))
            continue

        # ranking
        if ranking:
            sentences = rank(model, row_can['body'], n=n_sent)
        else:
            sentences = sent_tokenize(row_can['body'], language='slovene')

        vectors = model.encode(sentences, convert_to_numpy=True, batch_size=512)
        avg_vector = average_vectors(vectors)
        target_vectors.append(avg_vector)

    # compute cosine similarity
    similarity_matrix = cosine_similarity(source_vector.reshape(1, -1), target_vectors)
    similarity_matrix_reshaped = similarity_matrix[0]
    candidates['similarity'] = similarity_matrix_reshaped
    candidates.sort_values(by=['similarity'], ascending=False, inplace=True)

    return candidates