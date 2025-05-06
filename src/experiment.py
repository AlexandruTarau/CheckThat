import os
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
from rank_bm25 import BM25Okapi
import torch
import torch.nn as nn
from torch.nn import MarginRankingLoss
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from tqdm import tqdm
import json

from AIR.src.bert_for_reranking import BertForReRanking
from AIR.src.pairwise_dataset import PairwiseDataset

from transformers import logging
from transformers import get_linear_schedule_with_warmup

from torch.utils.tensorboard import SummaryWriter


# Suppress warnings from transformers
logging.set_verbosity_error()

def main():
    # === File Paths ===
    PATH_COLLECTION = 'subtask4b_collection_data.pkl'
    PATH_QUERY_TRAIN = 'subtask4b_query_tweets_train.tsv'
    PATH_QUERY_DEV = 'subtask4b_query_tweets_dev.tsv'

    # === TensorBoard Setup ===
    writer = SummaryWriter(log_dir="runs/re_rank_experiment")

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

    # === Load or Compute BM25 Top-1000 ===
    for split in ['train', 'dev']:
        pkl = f'bm25_top1000_{split}.pkl'
        if os.path.exists(pkl):
            print(f"Loading precomputed BM25 results for {split}...")
            df = pd.read_pickle(pkl)
            if split == 'train':
                df_query_train = df
            else:
                df_query_dev = df
        else:
            print(f"Computing BM25 top-1000 for {split}...")
            df = pd.read_csv(PATH_QUERY_TRAIN if split=='train' else PATH_QUERY_DEV, sep='\t')
            df['bm25_top1000'] = df['tweet_text'].progress_apply(
                lambda txt: bm25.get_top_n(preprocess(txt).split(), cord_uids, n=1000)
            )
            df.to_pickle(pkl)
            if split == 'train':
                df_query_train = df
            else:
                df_query_dev = df

        # Check how many true positives are in the top-1000
        found = sum(row['cord_uid'] in row['bm25_top1000'] for _, row in df.iterrows())
        print(f"Recall@1000: {found}/{len(df)} ({found / len(df):.2%})")

    print(f"Query train loaded: {df_query_train.shape}")
    print(f"Query dev loaded:   {df_query_dev.shape}")

    # === Build lookup for doc texts ===
    cord_to_text = dict(zip(df_collection['cord_uid'], df_collection['text_clean']))

    # === Load or Build Pairwise Dataset ===
    for split in ['train', 'dev']:
        pkl = f'pairwise_{split}.pkl'
        df_queries = df_query_train if split=='train' else df_query_dev
        if os.path.exists(pkl):
            print(f"Loading pairwise dataset for {split}...")
            if split == 'train':
                df_train_pw = pd.read_pickle(pkl)
            else:
                df_dev_pw = pd.read_pickle(pkl)
        else:
            print(f"Building pairwise dataset for {split} ({len(df_queries)} queries)...")
            data = []
            for _, row in tqdm(df_queries.iterrows(), total=len(df_queries), desc=f"Building PW {split}"):
                query = preprocess(row['tweet_text'])
                true_uid = row['cord_uid']
                if true_uid not in cord_to_text:
                    continue
                rel_doc = cord_to_text[true_uid]

                if true_uid not in row['bm25_top1000']:
                    print(f"Warning!! true_uid \"{true_uid}\" not in bm25_top1000.")
                    continue

                # Sample one non-relevant document
                non_relevants = []
                for cand in row['bm25_top1000']:
                    if cand in cord_to_text and cand != true_uid:
                        non_relevants.append(cand)
                if not non_relevants:
                    continue
                # Sample one non-relevant document Randomly
                # non_rel_doc = cord_to_text[np.random.choice(non_relevants)]

                # Hard negative sampling
                non_rel_doc = cord_to_text[non_relevants[0]]

                data.append({'query': query, 'rel_doc': rel_doc, 'non_rel_doc': non_rel_doc})

            df_pw = pd.DataFrame(data, columns=['query','rel_doc','non_rel_doc'])
            df_pw.to_pickle(pkl)
            if split=='train':
                df_train_pw = df_pw
            else:
                df_dev_pw = df_pw

    print(f"Pairwise train: {df_train_pw.shape}")
    print(f"Pairwise dev:   {df_dev_pw.shape}")

    # === Prepare DataLoaders ===
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Ensure sequences are padded/truncated properly
    train_dataset = PairwiseDataset(df_train_pw, tokenizer, max_length=256)
    dev_dataset = PairwiseDataset(df_dev_pw, tokenizer, max_length=256)

    # Updated DataLoader:
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=16)

    # === Model, Optimizer, Loss ===
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = BertForReRanking().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    loss_fn = MarginRankingLoss(margin=1.0)

    # === Training Loop with mixed precision & accumulation ===
    from torch.amp import GradScaler, autocast
    scaler = GradScaler()

    try:
        with open('rerank_model/metrics.json', 'r') as f:
            metrics = json.load(f)
            epoch_index = metrics['epochs']
            epochs = epoch_index + 3
            best_accuracy = metrics['best_accuracy']
            best_epoch = metrics['best_epoch']
    except FileNotFoundError:
        epoch_index = 0
        epochs = 3
        best_accuracy = 0.0
        best_epoch = 0

    accum_steps = 4  # accumulate gradients over 4 steps to simulate batch=32
    total_steps = len(train_loader) * epochs
    warmup_steps = total_steps // 10  # 10% of total steps

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    for epoch in range(1 + epoch_index, epochs+1):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]"), 1):
            # Each batch contains tokenized inputs for both relevant and non-relevant docs
            q_rel_input_ids = batch['rel_input_ids'].to(device)
            q_rel_attention = batch['rel_attention_mask'].to(device)
            q_non_rel_input_ids = batch['non_rel_input_ids'].to(device)
            q_non_rel_attention = batch['non_rel_attention_mask'].to(device)

            with autocast(device_type=('cuda' if torch.cuda.is_available() else 'cpu')):
                # Get scores (logits) for relevant and non-relevant pairs
                rel_scores = model(input_ids=q_rel_input_ids, attention_mask=q_rel_attention).view(-1)
                non_rel_scores = model(input_ids=q_non_rel_input_ids, attention_mask=q_non_rel_attention).view(-1)

                # MarginRankingLoss wants rel > non-rel => target is 1
                target = torch.ones(rel_scores.size(), device=device)
                loss = loss_fn(rel_scores, non_rel_scores, target) / accum_steps

            scaler.scale(loss).backward()

            if step % accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                # Update the learning rate
                scheduler.step()

                # Log the loss and learning rate
                global_step = (epoch - 1) * len(train_loader) + step
                writer.add_scalar("Loss/train", loss.item() * accum_steps, global_step)
                writer.flush()
                writer.add_scalar("LR", scheduler.get_last_lr()[0], global_step)
                writer.flush()

            total_loss += loss.item() * accum_steps

        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch} Train Loss: {avg_train_loss:.4f}")
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.flush()
        writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], epoch)
        writer.flush()

        # Evaluation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in tqdm(dev_loader, desc=f"Epoch {epoch}/{epochs} [Eval]"):
                rel_ids = batch['rel_input_ids'].to(device)
                rel_mask = batch['rel_attention_mask'].to(device)
                non_rel_ids = batch['non_rel_input_ids'].to(device)
                non_rel_mask = batch['non_rel_attention_mask'].to(device)

                rel_scores = model(input_ids=rel_ids, attention_mask=rel_mask).view(-1)
                non_rel_scores = model(input_ids=non_rel_ids, attention_mask=non_rel_mask).view(-1)

                correct += (rel_scores > non_rel_scores).sum().item()
                total += rel_scores.size(0)

        # Calculate accuracy
        acc = correct / total
        print(f"Epoch {epoch} Eval Accuracy: {acc:.4f}")
        writer.add_scalar('Accuracy/dev', acc, epoch)

        # Create directory for saving models
        os.makedirs('rerank_model', exist_ok=True)

        # Save the model
        model_path = f'rerank_model/reranker_epoch_{epoch}.pt'
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

        # === Save best model ===
        if acc > best_accuracy:
            best_accuracy = acc
            best_epoch = epoch
            best_model_path = 'rerank_model/reranker_best_model.pt'
            torch.save(model.state_dict(), best_model_path)
            print(f"[WOW!] New best model saved at epoch {epoch} with accuracy {acc:.4f}")

        # Save the best accuracy and epoch
        metrics = {
            'epochs': epoch,
            'best_accuracy': best_accuracy,
            'best_epoch': best_epoch,
        }
        with open('rerank_model/metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to rerank_model/metrics.json")

        # Log the evaluation results
        with open('rerank_model/eval_log.txt', 'a') as log_file:
            log_file.write(f"Epoch {epoch}, Accuracy: {acc:.4f}\n")

        if epoch == epoch_index + 1:
            # Save tokenizer once
            tokenizer.save_pretrained('rerank_model/tokenizer')
            print(f"Tokenizer saved to rerank_model/tokenizer")

            # Save training config once
            training_config = {
                'batch_size': 16,
                'learning_rate': 1e-5,
                'epochs': 3,
                'accum_steps': 4,
                'max_length': 256,
            }
            with open('rerank_model/config.json', 'w') as f:
                json.dump(training_config, f, indent=2)
            print(f"Training config saved to rerank_model/config.json")

    # Close TensorBoard writer
    writer.close()

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
