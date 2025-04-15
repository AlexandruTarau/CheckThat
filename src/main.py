import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize

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
    #print("Preprocessing:", text[:100])
    tokens = wordpunct_tokenize(text.lower())
    tokens = [w for w in tokens if w.isalpha() and w not in stop_words]
    return ' '.join(tokens)

# === Prepare Corpus ===
df_collection['text'] = df_collection['title'] + ' ' + df_collection['abstract']
df_collection['text_clean'] = df_collection['text'].apply(preprocess)

collection_texts = df_collection['text_clean'].tolist()
cord_uids = df_collection['cord_uid'].tolist()

# === Fit TF-IDF Vectorizer ===
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(collection_texts)

# === Function to Get Top-5 Predictions ===
def get_top5_predictions(tweet_text):
    query_clean = preprocess(tweet_text)
    query_vec = tfidf.transform([query_clean])
    cos_sim = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top5_idx = np.argsort(-cos_sim)[:5]
    return [cord_uids[i] for i in top5_idx]

# === Apply on Train and Dev ===
df_query_train['tfidf_top5'] = df_query_train['tweet_text'].apply(get_top5_predictions)
df_query_dev['tfidf_top5'] = df_query_dev['tweet_text'].apply(get_top5_predictions)

# === Evaluation Function ===
def get_performance_mrr(data, col_gold, col_pred, list_k=[1, 5, 10]):
    d_performance = {}
    for k in list_k:
        data['in_topx'] = data.apply(lambda x: (1/([i for i in x[col_pred][:k]].index(x[col_gold]) + 1) if x[col_gold] in x[col_pred][:k] else 0), axis=1)
        d_performance[k] = data['in_topx'].mean()
    return d_performance

# === Evaluate ===
results_train = get_performance_mrr(df_query_train, 'cord_uid', 'tfidf_top5')
results_dev = get_performance_mrr(df_query_dev, 'cord_uid', 'tfidf_top5')

print("Results on the train set:", results_train)
print("Results on the dev set:", results_dev)

# === Export Predictions ===
df_query_train[['post_id', 'tfidf_top5']].rename(columns={'tfidf_top5': 'preds'}).to_csv('predictions_train.tsv', sep='\t', index=False)
df_query_dev[['post_id', 'tfidf_top5']].rename(columns={'tfidf_top5': 'preds'}).to_csv('predictions_dev.tsv', sep='\t', index=False)
