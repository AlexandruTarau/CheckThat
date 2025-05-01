import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
from rank_bm25 import BM25Okapi
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from transformers import BertForSequenceClassification, BertTokenizer
import torch
import torch.nn as nn

from AIR.src.bert_for_reranking import BertForReRanking
from AIR.src.pointwise_dataset import PointwiseDataset

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


# Load pre-trained BERT for sequence classification (binary classification)
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

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

# === Function to Get Top-100 Predictions using BM25 ===
def get_top100_predictions_bm25(tweet_text):
    query_tokens = preprocess(tweet_text).split()
    scores = bm25.get_scores(query_tokens)
    top100_idx = np.argsort(-scores)[:100]
    return [cord_uids[i] for i in top100_idx]

# Try to load precomputed BM25 results
if os.path.exists('bm25_top100_train.pkl') and os.path.exists('bm25_top100_dev.pkl'):
    print("Loading precomputed BM25 results...", end=" ")
    df_query_train = pd.read_pickle('bm25_top100_train.pkl')
    df_query_dev = pd.read_pickle('bm25_top100_dev.pkl')
    print("Done.")
else:
    print("BM25 results not found. Running BM25 retrieval...")
    # [run your BM25 code to compute top100 candidates]
    df_query_train['bm25_top100'] = df_query_train['tweet_text'].progress_apply(get_top100_predictions_bm25)
    df_query_dev['bm25_top100'] = df_query_dev['tweet_text'].progress_apply(get_top100_predictions_bm25)
    # Save them for future runs
    df_query_train.to_pickle('bm25_top100_train.pkl')
    df_query_dev.to_pickle('bm25_top100_dev.pkl')
    print("Done.")


# === Build lookup from cord_uid to document text ===
cord_uid_to_text = dict(zip(df_collection['cord_uid'], df_collection['text_clean']))

# === Build Pointwise dataset ===
def build_pointwise_dataset(df_queries, top_col='bm25_top100'):
    data = []
    for _, row in df_queries.iterrows():
        query = preprocess(row['tweet_text'])
        true_doc = row['cord_uid']
        candidates = row[top_col]
        for candidate_uid in candidates:
            if candidate_uid in cord_uid_to_text:
                doc = cord_uid_to_text[candidate_uid]
                label = 1 if candidate_uid == true_doc else 0
                data.append((query, doc, label))
    return pd.DataFrame(data, columns=['query', 'doc', 'label'])


def train_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


# === Create datasets ===
df_train_pointwise = build_pointwise_dataset(df_query_train, top_col='bm25_top100')
df_dev_pointwise = build_pointwise_dataset(df_query_dev, top_col='bm25_top100')

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_dataset = PointwiseDataset(df_train_pointwise, tokenizer)
dev_dataset = PointwiseDataset(df_dev_pointwise, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=16, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BertForReRanking().to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
loss_fn = nn.CrossEntropyLoss()

# Train
for epoch in range(3):
    train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
    print(f"Epoch {epoch+1}: Train loss = {train_loss:.4f}")


"""
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
"""