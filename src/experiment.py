import os
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
from rank_bm25 import BM25Okapi
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from tqdm import tqdm
import json

from AIR.src.bert_for_reranking import BertForReRanking
from AIR.src.pointwise_dataset import PointwiseDataset

from transformers import logging

# Suppress warnings from transformers
logging.set_verbosity_error()

def main():
    # === File Paths ===
    PATH_COLLECTION = 'subtask4b_collection_data.pkl'
    PATH_QUERY_TRAIN = 'subtask4b_query_tweets_train.tsv'
    PATH_QUERY_DEV = 'subtask4b_query_tweets_dev.tsv'

    # === Load Collection ===
    df_collection = pd.read_pickle(PATH_COLLECTION)
    print(f"Collection loaded: {df_collection.shape}")

    # === Preprocessing ===
    stop_words = set(stopwords.words('english'))
    def preprocess(text):
        tokens = wordpunct_tokenize(text.lower())
        tokens = [w for w in tokens if w.isalpha() and w not in stop_words]
        return ' '.join(tokens)

    # === Prepare Corpus Text ===
    df_collection['text'] = df_collection['title'].fillna('') + ' ' + df_collection['abstract'].fillna('')
    df_collection['text_clean'] = df_collection['text'].apply(preprocess)

    # === BM25 Setup ===
    tqdm.pandas(desc="Initializing BM25")
    corpus_tokenized = [doc.split() for doc in df_collection['text_clean']]
    bm25 = BM25Okapi(corpus_tokenized)
    cord_uids = df_collection['cord_uid'].tolist()

    # === Load or Compute BM25 Top-100 ===
    for split in ['train', 'dev']:
        pkl = f'bm25_top100_{split}.pkl'
        if os.path.exists(pkl):
            print(f"Loading precomputed BM25 results for {split}...")
            if split == 'train':
                df_query_train = pd.read_pickle(pkl)
            else:
                df_query_dev = pd.read_pickle(pkl)
        else:
            print(f"Computing BM25 top-100 for {split}...")
            df = pd.read_csv(PATH_QUERY_TRAIN if split=='train' else PATH_QUERY_DEV, sep='\t')
            df['bm25_top100'] = df['tweet_text'].progress_apply(
                lambda txt: bm25.get_top_n(preprocess(txt).split(), cord_uids, n=100)
            )
            df.to_pickle(pkl)
            if split == 'train':
                df_query_train = df
            else:
                df_query_dev = df

    print(f"Query train loaded: {df_query_train.shape}")
    print(f"Query dev loaded:   {df_query_dev.shape}")

    # === Build lookup for doc texts ===
    cord_to_text = dict(zip(df_collection['cord_uid'], df_collection['text_clean']))

    # === Load or Build Pointwise Dataset ===
    for split in ['train', 'dev']:
        pkl = f'pointwise_{split}.pkl'
        df_queries = df_query_train if split=='train' else df_query_dev
        if os.path.exists(pkl):
            print(f"Loading pointwise dataset for {split}...")
            if split == 'train':
                df_train_pw = pd.read_pickle(pkl)
            else:
                df_dev_pw = pd.read_pickle(pkl)
        else:
            print(f"Building pointwise dataset for {split} ({len(df_queries)} queries)...")
            data = []
            for _, row in tqdm(df_queries.iterrows(), total=len(df_queries), desc=f"Building PW {split}"):
                query = preprocess(row['tweet_text'])
                true_uid = row['cord_uid']
                for cand in row['bm25_top100']:
                    if cand in cord_to_text:
                        label = int(cand == true_uid)
                        data.append((query, cord_to_text[cand], label))
            df_pw = pd.DataFrame(data, columns=['query','doc','label'])
            df_pw.to_pickle(pkl)
            if split=='train':
                df_train_pw = df_pw
            else:
                df_dev_pw = df_pw

    print(f"Pointwise train: {df_train_pw.shape}")
    print(f"Pointwise dev:   {df_dev_pw.shape}")

    # === Prepare DataLoaders ===
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Ensure sequences are padded/truncated properly
    train_dataset = PointwiseDataset(df_train_pw, tokenizer, max_length=256)
    dev_dataset = PointwiseDataset(df_dev_pw, tokenizer, max_length=256)

    # Updated DataLoader:
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=16)

    # === Model, Optimizer, Loss ===
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = BertForReRanking().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-7)
    loss_fn = nn.CrossEntropyLoss()

    # === Training Loop with mixed precision & accumulation ===
    from torch.amp import GradScaler, autocast
    scaler = GradScaler()
    epochs = 3
    accum_steps = 4  # accumulate gradients over 4 steps to simulate batch=32

    best_accuracy = 0.0

    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]"), 1):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            # === DEBUG #1: confirm device and shapes ===
            if step == 1:
                print(f"[DEBUG] input_ids.device: {input_ids.device}, labels.device: {labels.device}")
                print(f"[DEBUG] input_ids.shape: {input_ids.shape}, labels.shape: {labels.shape}")

            with autocast(device_type=('cuda' if torch.cuda.is_available() else 'cpu')):
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(logits, labels)

            # === DEBUG #2: inspect logits & loss ===
            if step == 1:
                print(f"[DEBUG] logits[:5]: {logits[:5].detach().cpu().numpy()}")
                print(f"[DEBUG] labels[:5]: {labels[:5].cpu().numpy()}")
                print(f"[DEBUG] initial loss: {loss.item():.6f}")

            scaler.scale(loss).backward()

            # === DEBUG #3: gradient norms ===
            #if step == 1:
            #    grad_norm = model.bert.embeddings.word_embeddings.weight.grad.norm().item()
            #    print(f"[DEBUG] grad norm (embeddings): {grad_norm:.6f}")

            if step % accum_steps == 0:
                # 1) unscale the gradients back to real values
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # 2) clip them and capture the pre-clip norm
                #grad_norm = torch.nn.utils.clip_grad_norm_(
                #    model.parameters(),
                #    max_norm = 1.0,
                #    norm_type = 2
                #)
                #print(f"[DEBUG] grad norm before clipping: {grad_norm:.6f}")

                # 3) step/update through the scaler
                scaler.step(optimizer)
                scaler.update()

                # 4) zero out for next accumulation
                optimizer.zero_grad()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch} Train Loss: {avg_train_loss:.4f}")

        # Evaluation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in tqdm(dev_loader, desc=f"Epoch {epoch}/{epochs} [Eval]"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        acc = correct / total
        print(f"Epoch {epoch} Eval Accuracy: {acc:.4f}")

        # Create directory for saving models
        os.makedirs('rerank_model', exist_ok=True)

        # Save the model
        model_path = f'rerank_model/reranker_epoch_{epoch}.pt'
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

        # === Save best model ===
        if acc > best_accuracy:
            best_accuracy = acc
            best_model_path = 'rerank_model/reranker_best_model.pt'
            torch.save(model.state_dict(), best_model_path)
            print(f"[WOW!] New best model saved at epoch {epoch} with accuracy {acc:.4f}")

        # Log the evaluation results
        with open('rerank_model/eval_log.txt', 'a') as log_file:
            log_file.write(f"Epoch {epoch}, Accuracy: {acc:.4f}\n")

        if epoch == 1:
            # Save tokenizer once
            tokenizer.save_pretrained('rerank_model/tokenizer')
            print(f"Tokenizer saved to rerank_model/tokenizer")

            # Save training config once
            training_config = {
                'batch_size': 16,
                'learning_rate': 5e-7,
                'epochs': 3,
                'accum_steps': 4,
                'max_length': 256,
            }
            os.makedirs('rerank_model', exist_ok=True)
            with open('rerank_model/config.json', 'w') as f:
                json.dump(training_config, f, indent=2)
            print(f"Training config saved to rerank_model/config.json")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
