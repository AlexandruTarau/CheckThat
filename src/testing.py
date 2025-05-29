import json
import os
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
from rank_bm25 import BM25Okapi
import torch
from transformers import BertTokenizer
from tqdm import tqdm
import warnings
import html
import re
import unicodedata

from bert_for_reranking import BertForReRanking

from transformers import logging
logging.set_verbosity_error()
warnings.filterwarnings("ignore")

# === Preprocessing ===
stop_words = set(stopwords.words('english'))
def preprocess_bm25(text):
    tokens = wordpunct_tokenize(text.lower())
    tokens = [w for w in tokens if w.isalpha() and w not in stop_words]
    return ' '.join(tokens)


def preprocess_reranker(text):
    text = html.unescape(text)
    text = unicodedata.normalize('NFKC', text)

    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)

    # Remove hashtags
    text = re.sub(r'#\w+', '', text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def get_performance_mrr(data, col_gold, col_pred, list_k=[1, 5, 10]):
    d_performance = {}
    for k in list_k:
        data['in_topx'] = data.apply(
            lambda x: (1 / ([i for i in x[col_pred][:k]].index(x[col_gold]) + 1)
                       if x[col_gold] in x[col_pred][:k] else 0),
            axis=1
        )
        d_performance[f"MRR@{k}"] = data['in_topx'].mean()
    return d_performance

def rerank_predictions(df_query, model, tokenizer, cord_to_text, device, df_collection):
    model.eval()
    results = []

    for _, row in tqdm(df_query.iterrows(), total=len(df_query), desc="Re-ranking"):
        query = preprocess_reranker(row['tweet_text'])
        # query = row['tweet_text']

        gold_uid = row['cord_uid']
        candidates = row['bm25_top5']

        inputs = []
        uid_list = []
        for cand_uid in candidates:
            if cand_uid not in cord_to_text:
                continue
            # doc = cord_to_text[cand_uid]
            doc = df_collection.loc[df_collection['cord_uid'] == cand_uid, 'text'].values[0]
            encoding = tokenizer(
                query, doc,
                max_length=256,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )
            inputs.append(encoding)
            uid_list.append(cand_uid)

        if not inputs:
            continue

        input_ids = torch.cat([i['input_ids'] for i in inputs]).to(device)
        attn_mask = torch.cat([i['attention_mask'] for i in inputs]).to(device)

        with torch.no_grad():
            scores = model(input_ids=input_ids, attention_mask=attn_mask).view(-1).cpu().numpy()

        sorted_uids = [uid for _, uid in sorted(zip(scores, uid_list), reverse=True)]
        results.append({'post_id': row['post_id'], 'cord_uid': gold_uid, 'reranked': sorted_uids})

    return pd.DataFrame(results)

def main():
    # === Load Configuration ===
    with open('config.json', 'r') as f:
        config = json.load(f)
        model_name = config.get('model_name', 'bert-base-uncased')
        # total_epochs = config.get('total_epochs', 10)
        # accum_steps = config.get('accum_steps', 1)
        # batch_size = config.get('batch_size', 16)
        # max_length = config.get('max_length', 256)
        # learning_rate = config.get('learning_rate', 1e-5)
        # bm25_top_k = config.get('bm25_top_k', 100)
        # log_dir = config.get('log_dir', 'runs/re_rank_experiment')
        save_dir = config.get('save_dir', 'rerank_model')
        PATH_COLLECTION = config.get('collection_path', 'subtask4b_collection_data.pkl')
        # PATH_QUERY_TRAIN = config.get('train_query_path', 'subtask4b_query_tweets_train.tsv')
        PATH_QUERY_DEV = config.get('dev_query_path', 'subtask4b_query_tweets_dev.tsv')

    # === Load Collection ===
    df_collection = pd.read_pickle(PATH_COLLECTION)
    df_collection['text'] = df_collection['title'].fillna('') + ' ' + df_collection['abstract'].fillna('')
    df_collection['text_clean'] = df_collection['text'].apply(preprocess_reranker)
    cord_to_text = dict(zip(df_collection['cord_uid'], df_collection['text']))

    # === BM25 Setup ===
    tqdm.pandas(desc="Initializing BM25")
    corpus_tokenized = [doc.split() for doc in df_collection['text_clean']]
    bm25 = BM25Okapi(corpus_tokenized)
    cord_uids = df_collection['cord_uid'].tolist()

    # === Load or Compute BM25 ===
    pkl = 'bm25_top5_dev.pkl'
    if os.path.exists(pkl):
        print(f"Loading precomputed BM25 results for dev...")
        df = pd.read_pickle(pkl)
        df_query_dev = df
    else:
        print(f"Computing BM25 for dev...")
        df = pd.read_csv(PATH_QUERY_DEV, sep='\t')
        df['bm25_top5'] = df['tweet_text'].progress_apply(
            lambda txt: bm25.get_top_n(preprocess_bm25(txt).split(), cord_uids, n=5)
        )
        df.to_pickle(pkl)
        df_query_dev = df

    # Check how many true positives are in the top-k
    found = sum(row['cord_uid'] in row['bm25_top5'] for _, row in df.iterrows())
    print(f"Recall@5: {found}/{len(df)} ({found / len(df):.2%})")

    print(f"Query dev loaded:   {df_query_dev.shape}")

    # === Tokenizer and Model ===
    tokenizer = BertTokenizer.from_pretrained(model_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load metrics
    try:
        with open(f'{save_dir}/metrics.json', 'r') as f:
            metrics = json.load(f)
        eval_epoch = metrics['eval_epoch']
        best_mrr = metrics['best_mrr']
        best_epoch = metrics['best_epoch']
    except FileNotFoundError:
        print("ERROR! Metrics file not found.")
        return

    # Evaluate the next model
    while True:
        eval_epoch += 1
        MODEL_PATH = f"{save_dir}/reranker_epoch_{eval_epoch}.pt"
        if os.path.exists(MODEL_PATH):
            print(f"Evaluating model from {MODEL_PATH}...")
            model = BertForReRanking().to(device)
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device)['model_state_dict'])
            df_dev_reranked = rerank_predictions(df_query_dev, model, tokenizer, cord_to_text, device, df_collection)
            mrr = get_performance_mrr(df_dev_reranked, col_gold='cord_uid', col_pred='reranked')
            print("MRR:", mrr)

            # Save the reranked predictions
            current_mrr1 = mrr.get('MRR@1', 0.0)
            current_mrr5 = mrr.get('MRR@5', 0.0)
            current_mrr10 = mrr.get('MRR@10', 0.0)

            # Update best_mrr and best_epoch if current is better
            if current_mrr10 > best_mrr[2]:
                best_mrr = [current_mrr1, current_mrr5, current_mrr10]
                best_epoch = eval_epoch
                print(f"New best model found at epoch {eval_epoch} with MRR: {mrr}")

            # Update metrics and save
            metrics['eval_epoch'] = eval_epoch
            metrics['best_mrr'] = best_mrr
            metrics['best_epoch'] = best_epoch

            with open(f'{save_dir}/metrics.json', 'w') as f:
                json.dump(metrics, f, indent=2)

            with open(f'{save_dir}/eval_log.txt', 'a') as log_file:
                log_file.write(f"Epoch {eval_epoch}, MRR: {mrr}\n")
        else:
            print(f"All models have been evaluated.")
            break

if __name__ == "__main__":
    main()
