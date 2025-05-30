
import numpy as np
import pandas as pd
import string
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import math
from collections import defaultdict, Counter

# Paths to data
PATH_COLLECTION_DATA = 'subtask4b_collection_data.pkl'
PATH_QUERY_DEV_DATA = 'subtask4b_query_tweets_dev.tsv'
PATH_QUERY_TRAIN_DATA = 'subtask4b_query_tweets_train.tsv'

# Load data
df_collection = pd.read_pickle(PATH_COLLECTION_DATA)
df_dev = pd.read_csv(PATH_QUERY_DEV_DATA, sep='\t')
df_train = pd.read_csv(PATH_QUERY_TRAIN_DATA, sep='\t')

corpus = df_collection[['title', 'abstract']].apply(lambda x: f"{x['title']} {x['abstract']}", axis=1).tolist()
cord_uids = df_collection['cord_uid'].tolist()

def preprocess(texts):
    return [
        [
            clean_word
            for word in doc.split()
            if (clean_word := word.strip(string.punctuation).lower()) and clean_word not in ENGLISH_STOP_WORDS
        ]
        for doc in texts
    ]

# Preprocess corpus
tokenized_corpus = preprocess(corpus)

def compute_tfidf(documents):
    N = len(documents)
    df = defaultdict(int)
    tfidf_matrix = []

    for doc in documents:
        for word in set(doc):
            df[word] += 1

    for doc in documents:
        tf = Counter(doc)
        doc_tfidf = {}
        for word, freq in tf.items():
            idf = math.log(N / (1 + df[word]))
            doc_tfidf[word] = freq * idf
        tfidf_matrix.append(doc_tfidf)

    return tfidf_matrix

corpus_tfidf = compute_tfidf(tokenized_corpus)

def cosine_sim(vec1, vec2):
    common = set(vec1) & set(vec2)
    num = sum(vec1[w] * vec2[w] for w in common)
    denom1 = math.sqrt(sum(v**2 for v in vec1.values()))
    denom2 = math.sqrt(sum(v**2 for v in vec2.values()))
    return num / (denom1 * denom2) if denom1 and denom2 else 0

def evaluate_query_set(df_query, name):
    tweets = df_query['tweet_text'].tolist()
    gold = df_query['cord_uid'].tolist()
    tokenized_tweets = preprocess(tweets)
    tweet_tfidf = compute_tfidf(tokenized_tweets)

    top_k = 10
    top_doc_ids = []

    for tweet_vector in tweet_tfidf:
        similarities = [(i, cosine_sim(tweet_vector, doc_vec)) for i, doc_vec in enumerate(corpus_tfidf)]
        top_indices = sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]
        top_docs = [cord_uids[i] for i, _ in top_indices]
        top_doc_ids.append(top_docs)

    mrr_at = {1: 0, 5: 0, 10: 0}
    for i, predictions in enumerate(top_doc_ids):
        for k in mrr_at.keys():
            if gold[i] in predictions[:k]:
                mrr_at[k] += 1 / (predictions.index(gold[i]) + 1)
                break

    total = len(gold)
    for k in mrr_at:
        mrr_at[k] /= total

    print(f"MRR@1 on {name}: {mrr_at[1]:.4f}")
    print(f"MRR@5 on {name}: {mrr_at[5]:.4f}")
    print(f"MRR@10 on {name}: {mrr_at[10]:.4f}")

# Run evaluation
evaluate_query_set(df_train, "train set")
evaluate_query_set(df_dev, "dev set")
