import json
import os
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi
import torch
from transformers import BertTokenizer
from tqdm import tqdm
import warnings
import html
import re
import unicodedata
from sentence_transformers import CrossEncoder
import nltk

from bert_for_reranking import BertForReRanking

from transformers import logging
logging.set_verbosity_error()
warnings.filterwarnings("ignore")

nltk.download('punkt_tab')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def preprocess_bm25(text):
    tokens = wordpunct_tokenize(text.lower())
    tokens = [w for w in tokens if w.isalpha() and w not in stop_words]
    return ' '.join(tokens)


def tokenize_for_bm25(text):
    tokens = word_tokenize(text.lower())
    return [t for t in tokens if t.isalpha()]


def preprocess_reranker(text):
    text = html.unescape(text)
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def get_top_cord_uids(text2bm25top, query, bm25, cord_uids, top_k):
    if query in text2bm25top:
        return text2bm25top[query]
    else:
        tokens = tokenize_for_bm25(query)
        tokens = [t for t in tokens if t.isalpha()]
        scores = bm25.get_scores(tokens)
        top_indices = np.argsort(-scores)[:top_k]
        top_cords = [cord_uids[i] for i in top_indices]
        text2bm25top[query] = top_cords
        return top_cords


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


def rerank_predictions(df_query, model, tokenizer, cord_to_text, device, df_collection, max_length, use_cross_encoder):
    model.eval()
    results = []

    for _, row in tqdm(df_query.iterrows(), total=len(df_query), desc="Re-ranking"):
        query = preprocess_reranker(row['tweet_text'])
        gold_uid = row['cord_uid']
        candidates = row['bm25_top5']

        inputs = []
        uid_list = []
        for cand_uid in candidates:
            if cand_uid not in cord_to_text:
                continue
            doc = df_collection.loc[df_collection['cord_uid'] == cand_uid, 'text'].values[0]
            if not use_cross_encoder:
                encoding = tokenizer(
                    query, doc,
                    max_length=max_length,
                    truncation=True,
                    padding='max_length',
                    return_tensors='pt'
                )
                inputs.append(encoding)
            uid_list.append(cand_uid)

        if not uid_list:
            continue

        if use_cross_encoder:
            pairs = [(query, cord_to_text[cand_uid]) for cand_uid in uid_list]
            scores = model.predict(pairs)
        else:
            input_ids = torch.cat([i['input_ids'] for i in inputs]).to(device)
            attn_mask = torch.cat([i['attention_mask'] for i in inputs]).to(device)
            with torch.no_grad():
                scores = model(input_ids=input_ids, attention_mask=attn_mask).view(-1).cpu().numpy()

        sorted_uids = [uid for _, uid in sorted(zip(scores, uid_list), reverse=True)]
        results.append({'post_id': row['post_id'], 'cord_uid': gold_uid, 'reranked': sorted_uids})

    return pd.DataFrame(results)


def main():
    with open('config.json', 'r') as f:
        config = json.load(f)
        model_name = config.get('model_name', 'bert-base-uncased')
        max_length = config.get('max_length', 256)
        save_dir = config.get('save_dir', 'rerank_model')
        PATH_COLLECTION = config.get('collection_path', 'subtask4b_collection_data.pkl')
        PATH_QUERY_TEST = config.get('test_query_path', 'subtask4b_query_tweets_test_gold.tsv')
        seed_everything(config.get("seed", 42))

    # === Prepare Corpus Text ===
    df_collection = pd.read_pickle(PATH_COLLECTION)
    df_collection['text'] = df_collection['title'].fillna('') + ' ' + df_collection['abstract'].fillna('')
    df_collection['text_clean'] = df_collection['text'].apply(preprocess_reranker)
    cord_to_text = dict(zip(df_collection['cord_uid'], df_collection['text']))

    # === BM25 Setup ===
    tqdm.pandas(desc="Initializing BM25")
    corpus_tokenized = [tokenize_for_bm25(doc) for doc in df_collection['text_clean']]
    bm25 = BM25Okapi(corpus_tokenized)
    cord_uids = df_collection['cord_uid'].tolist()

    pkl = 'bm25_top5_test.pkl'
    if os.path.exists(pkl):
        print(f"Loading precomputed BM25 results for testing...")
        df = pd.read_pickle(pkl)
        df_query_test = df
    else:
        print(f"Computing BM25 for test...")
        df = pd.read_csv(PATH_QUERY_TEST, sep='\t')
        text2bm25top = {}
        df['bm25_top5'] = df['tweet_text'].progress_apply(
            lambda txt: get_top_cord_uids(text2bm25top, txt, bm25, cord_uids, top_k=5)
        )
        df.to_pickle(pkl)
        df_query_test = df

    found = sum(row['cord_uid'] in row['bm25_top5'] for _, row in df.iterrows())
    print(f"Recall@5: {found}/{len(df)} ({found / len(df):.2%})")
    print(f"Query test loaded:   {df_query_test.shape}")

    tokenizer = BertTokenizer.from_pretrained(model_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Check if using pretrained CrossEncoder model
    use_cross_encoder = model_name in [
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "cross-encoder/ms-marco-MiniLM-L-12-v2",
    ]

    # Load metrics
    try:
        with open(f'{save_dir}/metrics.json', 'r') as f:
            metrics = json.load(f)
        eval_epoch = metrics.get('eval_epoch', 0)
        best_mrr = metrics.get('best_mrr', [0, 0, 0])
        best_epoch = metrics.get('best_epoch', 0)
    except FileNotFoundError:
        print("Metrics file not found, initializing new metrics.")
        eval_epoch = 0
        best_mrr = [0, 0, 0]
        best_epoch = 0
        metrics = {}

    # Otherwise, evaluate epoch-by-epoch using saved .pt models
    while True:
        eval_epoch += 1
        MODEL_PATH = f"{save_dir}/reranker_epoch_{eval_epoch}.pt"
        if not os.path.exists(MODEL_PATH):
            print("All saved models evaluated.")
            break

        print(f"Evaluating model from {MODEL_PATH}...")

        if use_cross_encoder:
            model = CrossEncoder(model_name, device=device)
            checkpoint = torch.load(MODEL_PATH, map_location=device)
            if 'model_state_dict' in checkpoint:
                # Strip 'model.' prefix from keys if present
                state_dict = checkpoint['model_state_dict']
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith('model.'):
                        new_key = k[len('model.'):]
                    else:
                        new_key = k
                    new_state_dict[new_key] = v
                model.model.load_state_dict(new_state_dict)
            else:
                model = checkpoint['model']
            model.to(device)
            model.eval()
        else:
            model = BertForReRanking().to(device)
            checkpoint = torch.load(MODEL_PATH, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()

        df_test_reranked = rerank_predictions(
            df_query_test, model, tokenizer, cord_to_text, device, df_collection, max_length, use_cross_encoder
        )

        # Save reranked top-5 predictions to TSV
        if not os.path.exists(f"{save_dir}/predictions"):
            os.makedirs(f"{save_dir}/predictions")
        output_tsv_path = f"{save_dir}/predictions/reranked_top5_epoch_{eval_epoch}.tsv"
        df_top5 = df_test_reranked[['post_id', 'reranked']].copy()

        # Extract top 5 predictions into separate columns
        for i in range(5):
            df_top5[f'top {i+1}'] = df_top5['reranked'].apply(lambda x: x[i] if len(x) > i else '')

        df_top5.drop(columns=['reranked'], inplace=True)
        df_top5.to_csv(output_tsv_path, sep='\t', index=False)
        print(f"Saved top-5 reranked predictions to {output_tsv_path}")

        # Get MRR@1, MRR@5, MRR@10
        mrr = get_performance_mrr(df_test_reranked, col_gold='cord_uid', col_pred='reranked')
        print("MRR:", mrr)

        current_mrr1 = mrr.get('MRR@1', 0.0)
        current_mrr5 = mrr.get('MRR@5', 0.0)
        current_mrr10 = mrr.get('MRR@10', 0.0)

        if current_mrr10 > best_mrr[2]:
            best_mrr = [current_mrr1, current_mrr5, current_mrr10]
            best_epoch = eval_epoch
            print(f"New best model found at epoch {eval_epoch} with MRR: {mrr}")

        metrics['eval_epoch'] = eval_epoch
        metrics['best_mrr'] = best_mrr
        metrics['best_epoch'] = best_epoch

        with open(f'{save_dir}/metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)

        with open(f'{save_dir}/eval_log.txt', 'a') as log_file:
            log_file.write(f"Epoch {eval_epoch}, MRR: {mrr}\n")


if __name__ == "__main__":
    main()
