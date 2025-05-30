import os
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
from nltk.tokenize import word_tokenize
import nltk
from rank_bm25 import BM25Okapi
import torch
from torch.nn import MarginRankingLoss
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from tqdm import tqdm
import json
from sentence_transformers import CrossEncoder

from bert_for_reranking import BertForReRanking
from pairwise_dataset import PairwiseDataset

from transformers import logging
from transformers import get_scheduler, get_cosine_schedule_with_warmup

import re
import html
import unicodedata

# Suppress warnings from transformers
logging.set_verbosity_error()

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

nltk.download('punkt_tab')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


# === Use the same seed for every random op for reproductibility ===
def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # For deterministic CNNs
    torch.backends.cudnn.benchmark = False     # Slower, but reproducible


# === Preprocessing ===
def tokenize_for_bm25(text):
    tokens = word_tokenize(text.lower())
    return [t for t in tokens if t.isalpha() and t not in stop_words]


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


def create_scheduler(optimizer, train_loader, total_epochs, accum_steps, scheduler_type):
    total_steps = (len(train_loader) // accum_steps) * total_epochs
    warmup_steps = int(total_steps * 0.1)

    if scheduler_type == "cosine":
        return get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
            num_cycles=0.5
        )
    else:
        return get_scheduler(
            name=scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )


def main():

    # === Load Configuration ===
    with open('config.json', 'r') as f:
        config = json.load(f)
        model_name = config.get('model_name', 'bert-base-uncased')
        total_epochs = config.get('total_epochs', 10)
        accum_steps = config.get('accum_steps', 1)
        batch_size = config.get('batch_size', 16)
        max_length = config.get('max_length', 256)
        learning_rate = config.get('learning_rate', 1e-5)
        bm25_top_k = config.get('bm25_top_k', 100)
        log_dir = config.get('log_dir', 'runs/re_rank_experiment')
        save_dir = config.get('save_dir', 'rerank_model')
        PATH_COLLECTION = config.get('collection_path', 'subtask4b_collection_data.pkl')
        PATH_QUERY_TRAIN = config.get('train_query_path', 'subtask4b_query_tweets_train.tsv')
        margin_loss = config.get('margin_loss', 1.0)
        weight_decay = config.get('weight_decay', 0.01)
        scheduler_type = config.get('scheduler', "linear")
        seed_everything(config.get("seed", 42))

    # === Save config ===
    with open(f'{save_dir}/config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # === Load Collection ===
    df_collection = pd.read_pickle(PATH_COLLECTION)
    print(f"Collection loaded: {df_collection.shape}")

    # === Prepare Corpus Text ===
    df_collection['text'] = df_collection['title'].fillna('') + ' ' + df_collection['abstract'].fillna('')
    df_collection['text_tokenized'] = df_collection['text'].apply(tokenize_for_bm25)

    # === BM25 Setup ===
    tqdm.pandas(desc="Initializing BM25")
    corpus_tokenized = df_collection['text_tokenized'].tolist()
    bm25 = BM25Okapi(corpus_tokenized)
    cord_uids = df_collection['cord_uid'].tolist()

    # === Load or Compute BM25 ===
    text2bm25top = {}
    pkl = f'bm25_top{bm25_top_k}_train.pkl'
    if os.path.exists(pkl):
        print("Loading precomputed BM25 results for train...")
        df_query_train = pd.read_pickle(pkl)
    else:
        print(f"Computing BM25 top-{bm25_top_k} for train...")
        df = pd.read_csv(PATH_QUERY_TRAIN, sep='\t')
        df[f'bm25_top{bm25_top_k}'] = df['tweet_text'].progress_apply(
            lambda txt: get_top_cord_uids(text2bm25top, txt, bm25, cord_uids, bm25_top_k)
        )
        df.to_pickle(pkl)
        df_query_train = df

    found = sum(row['cord_uid'] in row[f'bm25_top{bm25_top_k}'] for _, row in df_query_train.iterrows())
    print(f"Recall@{bm25_top_k} (train): {found}/{len(df_query_train)} ({found / len(df_query_train):.2%})")
    print(f"Query train loaded: {df_query_train.shape}")

    # === Build lookup for doc texts ===
    cord_to_text = dict(zip(df_collection['cord_uid'], df_collection['text']))

    # === Load or Build Pairwise Dataset ===
    pkl = 'pairwise_train.pkl'
    if os.path.exists(pkl):
        print("Loading pairwise dataset for train...")
        df_train_pw = pd.read_pickle(pkl)
    else:
        print(f"Building pairwise dataset for train ({len(df_query_train)} queries)...")
        data = []
        for _, row in tqdm(df_query_train.iterrows(), total=len(df_query_train), desc="Building PW train"):
            query = preprocess_reranker(row['tweet_text'])
            true_uid = row['cord_uid']
            if true_uid not in cord_to_text:
                continue
            rel_doc = df_collection.loc[df_collection['cord_uid'] == true_uid, 'text'].values[0]
            if true_uid not in row[f'bm25_top{bm25_top_k}']:
                print(f"Warning!! true_uid \"{true_uid}\" not in bm25_top{bm25_top_k}.")
                continue
            non_relevants = [cand for cand in row[f'bm25_top{bm25_top_k}'] if cand in cord_to_text and cand != true_uid]
            if not non_relevants:
                continue
            hard_negatives = non_relevants[:10]
            if not hard_negatives:
                continue
            chosen = np.random.choice(hard_negatives) if len(hard_negatives) > 1 else hard_negatives[0]
            non_rel_doc = df_collection.loc[df_collection['cord_uid'] == chosen, 'text'].values[0]
            data.append({'query': query, 'rel_doc': rel_doc, 'non_rel_doc': non_rel_doc})
        df_train_pw = pd.DataFrame(data, columns=['query', 'rel_doc', 'non_rel_doc'])
        df_train_pw.to_pickle(pkl)

    print(f"Pairwise train: {df_train_pw.shape}")

    # === Prepare DataLoaders ===
    tokenizer = BertTokenizer.from_pretrained(model_name)
    train_dataset = PairwiseDataset(df_train_pw, tokenizer, max_length=max_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # === Model Setup ===
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_cross_encoder = model_name in [
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "cross-encoder/ms-marco-MiniLM-L-12-v2",
    ]

    if use_cross_encoder:
        model = CrossEncoder(model_name, device=device)
        optimizer = torch.optim.AdamW(model.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        model = BertForReRanking().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = MarginRankingLoss(margin=margin_loss)

    # === Mixed Precision Setup ===
    from torch.amp import GradScaler, autocast
    scaler = GradScaler()

    # === Checkpoint Loading ===
    checkpoint_path = f'{save_dir}/checkpoint_latest.pt'
    if os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1

        # === Scheduler Setup ===
        scheduler = create_scheduler(optimizer, train_loader, total_epochs, accum_steps, scheduler_type)
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    else:
        print("No checkpoint found. Starting from scratch.")
        start_epoch = 1

        # === Scheduler Setup ===
        scheduler = create_scheduler(optimizer, train_loader, total_epochs, accum_steps, scheduler_type)

    # === Training Loop ===
    for epoch in range(start_epoch, total_epochs + 1):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}/{total_epochs} [Train]"), 1):
            q_rel_input_ids = batch['rel_input_ids'].to(device)
            q_rel_attention = batch['rel_attention_mask'].to(device)
            q_non_rel_input_ids = batch['non_rel_input_ids'].to(device)
            q_non_rel_attention = batch['non_rel_attention_mask'].to(device)

            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                if use_cross_encoder:
                    # Prepare input pairs for CrossEncoder
                    sentence_pairs = list(zip(batch['query'], batch['rel_doc'])) + list(zip(batch['query'], batch['non_rel_doc']))
                    inputs = model.tokenizer(sentence_pairs, padding=True, truncation=True, return_tensors='pt').to(device)

                    outputs = model.model(**inputs)  # forward pass, logits with grad
                    logits = outputs.logits.view(-1)

                    rel_scores = logits[:len(batch['query'])]
                    non_rel_scores = logits[len(batch['query']):]
                else:
                    rel_scores = model(input_ids=q_rel_input_ids, attention_mask=q_rel_attention).view(-1)
                    non_rel_scores = model(input_ids=q_non_rel_input_ids, attention_mask=q_non_rel_attention).view(-1)

                target = torch.ones_like(rel_scores)
                loss = loss_fn(rel_scores, non_rel_scores, target) / accum_steps

            scaler.scale(loss).backward()

            if step % accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scheduler.step()
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item() * accum_steps

        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch} Train Loss: {avg_train_loss:.4f}")

        # Save loss, learning rate in train_log.txt
        with open(f'{save_dir}/train_log.txt', 'a') as log_file:
            current_lr = scheduler.get_last_lr()[0]
            log_file.write(f"Epoch {epoch}, Loss: {avg_train_loss:.4f}, LR: {current_lr:.6f}\n")

        # Save model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
        }, f'{save_dir}/reranker_epoch_{epoch}.pt')

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
        }, f'{save_dir}/checkpoint_latest.pt')


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()