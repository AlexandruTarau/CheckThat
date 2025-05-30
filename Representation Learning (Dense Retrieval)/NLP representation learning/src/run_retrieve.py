#!/usr/bin/env python3
import json
import pathlib
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


# set paths
tree_root  = pathlib.Path(__file__).parent.parent # change maybe if it makes problems 
data_dir   = tree_root / "data"
models_dir = tree_root / "models"
run_dir    = tree_root

# select latest model checkpoint
model_dir = sorted(models_dir.glob("dense_*"))[-1]

# load model
print(f"-> loading model from {model_dir}")
model = SentenceTransformer(str(model_dir))

# load faiss index
index_file = models_dir / "paper_index.faiss"
print(f"-> loading faiss index from {index_file}")
index = faiss.read_index(str(index_file))

# load paper ids
paper_ids = pd.read_pickle(models_dir / "paper_ids.pkl")["cord_uid"].tolist()

# load train and dev queries
train_df = pd.read_csv(
    data_dir / "subtask4b_query_tweets_train.tsv",
    sep="\t",
    names=["post_id", "tweet", "cord_uid"],
    dtype={"post_id": str}
)
dev_df = pd.read_csv(
    data_dir / "subtask4b_query_tweets_dev.tsv",
    sep="\t",
    names=["post_id", "tweet", "cord_uid"],
    dtype={"post_id": str}
)
print(f"-> loaded {len(train_df)} train and {len(dev_df)} dev queries")
all_queries = pd.concat([train_df, dev_df], ignore_index=True)

# encode tweets (no background workers)
print(f"-> encoding {len(all_queries)} tweets")
embeddings = model.encode(
    all_queries.tweet.tolist(),
    batch_size=64,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True,
    num_workers=0      # <— force single‐process encoding
)

# retrieve top-k for each query
k = 5
batch_size = 512
predictions = []
for start in tqdm(range(0, len(embeddings), batch_size)):
    batch = embeddings[start : start + batch_size].astype(np.float32)
    _, indices = index.search(batch, k)
    predictions.extend(indices.tolist())

# split train and dev results
n_train     = len(train_df)
train_preds = predictions[:n_train]
dev_preds   = predictions[n_train:]

# write train-only predictions
out_train = run_dir / "dense_train_only.tsv"
with out_train.open("w") as f:
    for pid, idxs in zip(train_df.post_id, train_preds):
        f.write(f"{pid}\t{json.dumps([paper_ids[i] for i in idxs])}\n")
print(f"train predictions saved to {out_train}")

# write dev-only predictions
out_dev = run_dir / "dense_dev_only.tsv"
with out_dev.open("w") as f:
    for pid, idxs in zip(dev_df.post_id, dev_preds):
        f.write(f"{pid}\t{json.dumps([paper_ids[i] for i in idxs])}\n")
print(f"dev predictions saved to {out_dev}")
