import numpy as np
import pandas as pd
import string
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import math
from collections import defaultdict, Counter


PATH_COLLECTION_DATA = 'subtask4b_collection_data.pkl'
df_collection = pd.read_pickle(PATH_COLLECTION_DATA)


PATH_QUERY_TEST_DATA = 'subtask4b_query_tweets_test.tsv'
df_query_test = pd.read_csv(PATH_QUERY_TEST_DATA, sep = '\t')


corpus = df_collection[:][['title', 'abstract']].apply(lambda x: f"{x['title']} {x['abstract']}", axis=1).tolist()
cord_uids = df_collection[:]['cord_uid'].tolist()

tokenized_corpus = [doc.split(' ') for doc in corpus]
tokenized_tweets = [doc.split(' ') for doc in df_query_test['tweet_text']]

def remove_stopwords(x):
    x = [
        [
            clean_word
            for word in sentence
            if (clean_word := word.strip(string.punctuation))
               and clean_word.lower() not in ENGLISH_STOP_WORDS
        ]
        for sentence in x]
    return x

removed_stopwords_tweets = remove_stopwords(tokenized_tweets)
removed_stopwords_corpus = remove_stopwords(tokenized_corpus)

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

corpus_tfidf = compute_tfidf(removed_stopwords_corpus)
tweet_tfidf = compute_tfidf(removed_stopwords_tweets)

def cosine_sim(vec1, vec2):
    common = set(vec1) & set(vec2)
    num = sum(vec1[w] * vec2[w] for w in common)
    denom1 = math.sqrt(sum(v**2 for v in vec1.values()))
    denom2 = math.sqrt(sum(v**2 for v in vec2.values()))
    return num / (denom1 * denom2) if denom1 and denom2 else 0

top_k = 5
cord_uids = df_collection['cord_uid'].tolist()
top_doc_ids = []
all_similarities = []

for tweet_vectors in tweet_tfidf:
    similarities = [(i, cosine_sim(tweet_vectors, doc_vec)) for i, doc_vec in enumerate(corpus_tfidf)]
    all_similarities.append(similarities)
    top_indices = sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]
    top_docs = [cord_uids[i] for i, _ in top_indices]
    top_doc_ids.append(top_docs)

similarity_threshold = 0.1 #The lower the threshold, the more tweets are included in query
true_relevant = {}
for i, sims in enumerate(all_similarities):
    relevant_doc_indices = [j for j, score in sims if score > similarity_threshold]
    if relevant_doc_indices:
        true_relevant[i] = {cord_uids[j] for j in relevant_doc_indices}

mrr = 0
count = 0
for i, top_docs in enumerate(top_doc_ids):
    if i not in true_relevant:
        continue
    for rank, doc in enumerate(top_docs, start=1):
        if doc in true_relevant[i]:
            mrr += 1 / rank
            break
    else:
        mrr += 0
    count += 1

mrr_score = mrr / count if count > 0 else 0
print(f"Cosine similarity based Top-5 has a MRR score: {mrr_score:.4f} ({count} evaluated tweets)")

post_ids = df_query_test['post_id'].tolist()
df_output = pd.DataFrame({
    "post_id": post_ids,
    "preds": [str(pred) for pred in top_doc_ids]
})
# Optional to save all the data in csv file, it is not asked, but we can do it as extra
#df_output.to_csv('output_predictions_tfidf.csv', sep='\t', index=False)

pd.set_option("display.max_colwidth", None)
print(df_output.head())

"""
So what is happening in this script; first the data is being collected from the data set and put into a dataframe. 
Then the dataframe is being tokenized and removed of stop words. After that the tfidf has been calculated and the results are being displayed.
At last the cosine similarity (cosine_sim) is being used to calculated the MRR score.
"""
