import numpy as np
import pandas as pd
import string
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from collections import defaultdict

PATH_COLLECTION_DATA = 'subtask4b_collection_data.pkl'

df_collection = pd.read_pickle(PATH_COLLECTION_DATA)
df_collection.info()
df_collection.head()

PATH_QUERY_TEST_DATA = 'subtask4b_query_tweets_test.tsv'
df_query_test = pd.read_csv(PATH_QUERY_TEST_DATA, sep = '\t')
df_query_test.head()
df_query_test.info()
df_query_test.head()

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

def generate_posting_list(x):
    posting_list = defaultdict(set)
    for doc_id, words in enumerate(x):
        for word in words:
            posting_list[word].add(doc_id)

    return {word: sorted(list(doc_ids)) for word, doc_ids in posting_list.items()}

posting_list_tweets = generate_posting_list(removed_stopwords_tweets)
posting_list_corpus = generate_posting_list(removed_stopwords_corpus)

#for word, docs in posting_list_tweets.items():
#    print(f"{word}: {docs}")

#print(posting_list_tweets)
print(posting_list_corpus)
