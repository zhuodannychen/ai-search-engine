import pandas as pd
import numpy as np
import os
import time
from sentence_transformers import SentenceTransformer
import faiss
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer, InputExample, losses, models, datasets, CrossEncoder 
from torch import nn
import os
import random


def create_embedding(model, index_name):
    print('Creating embeddings...')
    embeddings = np.array([model.encode(text) for text in df['lemmatized_question_body']]).astype('float32')
    # Create a FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, f'{PATH_TO_DATA}{index_name}')
    return index

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

def count_tokens(text):
    return len(text.split())

def create_pretrained_model():
    pretrained_model = SentenceTransformer('multi-qa-mpnet-base-cos-v1')
    pretrained_index = None
    if not os.path.exists(f'{PATH_TO_DATA}pretrained_index.faiss'):
        pretrained_index = create_embedding(pretrained_model, 'pretrained_index.faiss')
    else:
        pretrained_index = faiss.read_index(f'{PATH_TO_DATA}pretrained_index.faiss')
    return pretrained_model, pretrained_index
    
def eval_model(model, index, query):
    index_results = search(query, index, model)
    for result in index_results:
        print()
        print(result['question_title'])
        print(result['question_url'])
    return index_results



def create_fine_tune_model():
    print("Creating fine-tuned model...")
    tokenizer = T5Tokenizer.from_pretrained('BeIR/query-gen-msmarco-t5-large-v1')
    syn_query_gen_model = T5ForConditionalGeneration.from_pretrained('BeIR/query-gen-msmarco-t5-large-v1')
    syn_query_gen_model.eval()

    small_df = df.sample(n=1000, random_state=42)
    print("Small df shape:", small_df.shape)

    def _removeNonAscii(s):
        return "".join(i for i in s if ord(i) < 128)

    device = 'cuda'
    posts = small_df['question_body'].tolist()
    batch_size = 16
    num_queries = 3 # num queries per post
    max_length_post = 512
    max_length_query = 64

    syn_query_gen_model.to(device)

    print("Generating queries post pair...")
    with open(f'{PATH_TO_DATA}gen_queries.tsv', 'w') as file_out:
        for start_idx in range(0, len(posts), batch_size):
            cur_post_batch = posts[start_idx:start_idx+batch_size]
            inputs = tokenizer.prepare_seq2seq_batch(cur_post_batch, max_length=max_length_post, truncation=True, return_tensors='pt').to(device)
            outputs = syn_query_gen_model.generate(
                **inputs,
                max_length=max_length_query,
                do_sample=True,
                top_p=0.95,
                num_return_sequences=num_queries)

            for idx, output in enumerate(outputs):
                query = tokenizer.decode(output, skip_special_tokens=True)
                query = _removeNonAscii(query)
                post = cur_post_batch[int(idx/num_queries)]
                post = _removeNonAscii(post)
                file_out.write("{}\t{}\n".format(query.replace("\t", " ").strip(), post.replace("\t", " ").strip()))

    train_samples = [] 
    with open(f'{PATH_TO_DATA}gen_queries.tsv') as file_in:
        for line in file_in:
            try:
                query, post = line.strip().split('\t')
                train_samples.append(InputExample(texts=[query, post]))
            except:
                pass
            
    random.shuffle(train_samples)

    print("Training model...")
    train_dataloader = datasets.NoDuplicatesDataLoader(train_samples, batch_size=8)

    word_embedding = models.Transformer('sentence-transformers/multi-qa-mpnet-base-cos-v1')
    pooling = models.Pooling(word_embedding.get_word_embedding_dimension())
    model = SentenceTransformer(modules=[word_embedding, pooling])

    # loss
    train_loss = losses.MultipleNegativesRankingLoss(model)

    # tune the model
    num_epochs = 3
    warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)
    model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=num_epochs, warmup_steps=warmup_steps, show_progress_bar=True)

    os.makedirs(f'{PATH_TO_DATA}models', exist_ok=True)
    model.save(f'{PATH_TO_DATA}models/fine-tuned-model')

    tuned_model = SentenceTransformer(f'{PATH_TO_DATA}models/fine-tuned-model')
    tuned_index = create_embedding(tuned_model, "fine-tuned-index.faiss")

    return tuned_model, tuned_index


def cross_score(model_inputs):
    scores = cross_model.predict(model_inputs)
    return scores



PATH_TO_DATA = '../../data/'

if __name__ == '__main__':
    df = pd.read_csv(f'{PATH_TO_DATA}stackoverflow-100000')
    print(df.shape)
    df['token_count'] = df['question_body'].apply(count_tokens)
    df = df[df['token_count'] <= 512]
    print(df.shape)

    pretrained_model, pretrained_index = create_pretrained_model()
    query = "how to sort integers in python"
    eval_model(pretrained_model, pretrained_index, query)

    print()
    print("++++++++++++++++++++++++++++++++++++++++++++++++")
    print()

    if not os.path.exists(f'{PATH_TO_DATA}fine-tuned-index.faiss'):
        tuned_model, tuned_index = create_fine_tune_model()
    else:
        tuned_model = SentenceTransformer(f'{PATH_TO_DATA}models/fine-tuned-model')
        tuned_index = faiss.read_index(f'{PATH_TO_DATA}fine-tuned-index.faiss')

    tuned_results = eval_model(tuned_model, tuned_index, query)

    # rerank using cross-encoders
    cross_model = CrossEncoder('cross-encoder/ms-marco-TinyBERT-L-6', max_length=512)
    model_inputs = [[query, item['question_body']] for item in tuned_results]
    scores = cross_score(model_inputs)
    ce_ranked_results = [{'question_title': inp['question_title'], 'question_url': inp['question_url'], 'cross_score': score} for inp, score in zip(tuned_results, scores)]
    ce_ranked_results = sorted(ce_ranked_results, key=lambda x: x['cross_score'], reverse=True)

    for result in ce_ranked_results:
        print()
        print(result['question_title'])
        print(result['question_url'])
        print(result['cross_score'])
