# src/run_retrieve.py

import json
import pathlib
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# -------------------------------------------------------------------
# 1.  Set paths
# -------------------------------------------------------------------
ROOT   = pathlib.Path("/Users/agonsylejmani/Downloads/AIR/AIR#/NLP representation learning")
DATA   = ROOT / "data"
MODELS = ROOT / "models"

# -------------------------------------------------------------------
# 2.  Load model and FAISS index
# -------------------------------------------------------------------
model_dir = sorted(MODELS.glob("dense_*"))[-1]  # use most recent model directory
model = SentenceTransformer(str(model_dir))

index = faiss.read_index(str(MODELS / "paper_index.faiss"))
paper_ids = pd.read_pickle(str(MODELS / "paper_ids.pkl"))["cord_uid"].tolist()

# -------------------------------------------------------------------
# 3.  Load and concatenate both train and dev queries
# -------------------------------------------------------------------
train = pd.read_csv(DATA / "subtask4b_query_tweets_train.tsv", sep="\t", names=["post_id", "tweet", "label"])
dev   = pd.read_csv(DATA / "subtask4b_query_tweets_dev.tsv", sep="\t", names=["post_id", "tweet", "label"])

all_queries = pd.concat([train, dev], ignore_index=True)

# -------------------------------------------------------------------
# 4.  Encode tweets and perform dense retrieval
# -------------------------------------------------------------------
q_emb = model.encode(
    all_queries.tweet.tolist(),
    batch_size=64,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)

all_preds = []
batch_size = 512
for i in tqdm(range(0, len(q_emb), batch_size)):
    batch = q_emb[i:i + batch_size].astype(np.float32)
    _, I = index.search(batch, k=5)
    all_preds.extend(I)

# -------------------------------------------------------------------
# 5.  Write predictions to TSV
# -------------------------------------------------------------------
run_path = ROOT / "dense_submission.tsv"
with run_path.open("w") as f:
    for pid, top5 in zip(all_queries.post_id, I):
        preds = [paper_ids[i] for i in top5]
        f.write(f"{pid}\t{json.dumps(preds)}\n")

print("✅ Wrote:", run_path)
