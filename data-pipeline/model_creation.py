import pandas as pd
import numpy as np
import os
import time
from tqdm import tqdm
import seaborn as sns
from textblob import TextBlob
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
import faiss


def create_embedding(column_name, model):
    df[column_name] = df['lemmatized_question_body'].apply(lambda x: model.encode(x))
    embeddings = np.array(df[column_name].tolist()).astype('float32')
    # Create a FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])  # Using L2 distance for similarity measure
    index.add(embeddings)
    return index


def fetch_post_info(df_idx):
    info = df.iloc[df_idx]
    meta_info = {
        'question_title': info['question_title'],
        'question_body': info['question_body'],
        'question_score': info['question_score']
    }
    return meta_info

def search(query, index, model, top_k=10):
    start_time = time.time()
    query_embedding = model.encode([query])
    search_results = index.search(query_embedding, top_k) # returns [[distance], [index]]
    end_time = time.time()
    print(f'Total search time: {end_time - start_time:.4f} seconds')

    # print(search_results)
    top_k_idx = np.unique(search_results[1]).tolist()
    top_k_posts = []
    for idx in top_k_idx:
        top_k_posts.append(fetch_post_info(idx))
    return top_k_posts


pretrained_model = SentenceTransformer('multi-qa-mpnet-base-cos-v1')
# pretrained_index = create_embedding('basic-embedding', pretrained_model)
pretrained_index = faiss.read_index('pretrained_index.faiss')

query = "how to sort integers in python"
search(query, pretrained_index, pretrained_model)