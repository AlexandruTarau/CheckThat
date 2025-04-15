import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
from rank_bm25 import BM25Okapi
from tqdm import tqdm

tqdm.pandas()

# === File Paths ===
PATH_COLLECTION_DATA = 'subtask4b_collection_data.pkl'
PATH_QUERY_TRAIN_DATA = 'subtask4b_query_tweets_train.tsv'
PATH_QUERY_DEV_DATA = 'subtask4b_query_tweets_dev.tsv'

# === Load Data ===
df_collection = pd.read_pickle(PATH_COLLECTION_DATA)
print("Collection data loaded:", df_collection.shape)
df_query_train = pd.read_csv(PATH_QUERY_TRAIN_DATA, sep='\t')
print("Query train data loaded:", df_query_train.shape)
df_query_dev = pd.read_csv(PATH_QUERY_DEV_DATA, sep='\t')
print("Query dev data loaded:", df_query_dev.shape)

# === Preprocessing Function ===
stop_words = set(stopwords.words('english'))
def preprocess(text):
    tokens = wordpunct_tokenize(text.lower())
    tokens = [w for w in tokens if w.isalpha() and w not in stop_words]
    return ' '.join(tokens)

# === Prepare Corpus ===
df_collection['text'] = df_collection['title'] + ' ' + df_collection['abstract']
df_collection['text_clean'] = df_collection['text'].apply(preprocess)

collection_texts = df_collection['text_clean'].tolist()
cord_uids = df_collection['cord_uid'].tolist()

# === BM25 === Tokenize the preprocessed text (space-separated string -> list of tokens)
corpus_tokenized = [doc.split() for doc in df_collection['text_clean']]
bm25 = BM25Okapi(corpus_tokenized)

# === Function to Get Top-5 Predictions using BM25 ===
def get_top5_predictions_bm25(tweet_text):
    query_tokens = preprocess(tweet_text).split()
    scores = bm25.get_scores(query_tokens)
    top5_idx = np.argsort(-scores)[:5]
    return [cord_uids[i] for i in top5_idx]


# === Apply on Train and Dev using BM25 ===
df_query_train['bm25_top5'] = df_query_train['tweet_text'].progress_apply(get_top5_predictions_bm25)
df_query_dev['bm25_top5'] = df_query_dev['tweet_text'].progress_apply(get_top5_predictions_bm25)


# === Evaluation Function ===
def get_performance_mrr(data, col_gold, col_pred, list_k=[1, 5, 10]):
    d_performance = {}
    for k in list_k:
        data['in_topx'] = data.apply(lambda x: (1/([i for i in x[col_pred][:k]].index(x[col_gold]) + 1) if x[col_gold] in x[col_pred][:k] else 0), axis=1)
        d_performance[k] = data['in_topx'].mean()
    return d_performance

# === Evaluate BM25 Predictions ===
results_train_bm25 = get_performance_mrr(df_query_train, 'cord_uid', 'bm25_top5')
results_dev_bm25 = get_performance_mrr(df_query_dev, 'cord_uid', 'bm25_top5')

print("BM25 Results on the train set:", results_train_bm25)
print("BM25 Results on the dev set:", results_dev_bm25)

# === Export Predictions for BM25 ===
df_query_train[['post_id', 'bm25_top5']].rename(columns={'bm25_top5': 'preds'}).to_csv('predictions_train.tsv', sep='\t', index=False)
df_query_dev[['post_id', 'bm25_top5']].rename(columns={'bm25_top5': 'preds'}).to_csv('predictions_dev.tsv', sep='\t', index=False)
