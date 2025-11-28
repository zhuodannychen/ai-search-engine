from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import time
from sentence_transformers import SentenceTransformer
import faiss
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer, InputExample, losses, models, datasets, CrossEncoder 
from torch import nn
import os
import random

app = Flask(__name__)
PATH_TO_DATA = "../../data/"

@app.route('/')
def home():
    return "Welcome to the simple Flask API!"

def fetch_post_info(df_idx):
    info = df.iloc[df_idx]
    meta_info = {
        'question_title': info['question_title'],
        'question_body': info['question_body'],
        'question_score': info['question_score'],
        'question_url': f"https://stackoverflow.com/questions/{info['question_id']}"
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

@app.route('/search', methods=['GET'])
def searchQuery():
    query = request.args.get('query', default=None, type=str)
    print(query)

    tuned_model = SentenceTransformer(f'{PATH_TO_DATA}models/fine-tuned-model')
    tuned_index = faiss.read_index(f'{PATH_TO_DATA}fine-tuned-index.faiss')

    top_k_posts = search(query, tuned_index, tuned_model)

    return top_k_posts


if __name__ == '__main__':
    df = pd.read_csv(f'{PATH_TO_DATA}stackoverflow-100000')
    app.run(port=8000, debug=True)