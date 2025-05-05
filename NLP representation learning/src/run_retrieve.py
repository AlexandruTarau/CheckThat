# src/run_retrieve.py
import json, pathlib, numpy as np
import pandas as pd, faiss, tqdm
from sentence_transformers import SentenceTransformer

# -------------------------------------------------------------------
# 1.  Paths
# -------------------------------------------------------------------
ROOT   = pathlib.Path("/Users/agonsylejmani/Downloads/AIR/AIR#/NLP representation learning")
DATA   = ROOT / "data"
MODELS = ROOT / "models"

# -------------------------------------------------------------------
# 2.  Load model and FAISS index
# -------------------------------------------------------------------
model_dir = sorted(MODELS.glob("dense_*"))[-1]        # newest checkpoint
model = SentenceTransformer(str(model_dir))

index = faiss.read_index(str(MODELS / "paper_index.faiss"))
paper_ids = pd.read_pickle(str(MODELS / "paper_ids.pkl"))["cord_uid"].tolist()

# -------------------------------------------------------------------
# 3.  Encode dev tweets and search top‑5 papers
# -------------------------------------------------------------------
dev = pd.read_csv(
    DATA / "subtask4b_query_tweets_dev.tsv",
    sep="\t",
    names=["post_id", "tweet", "label"]
)

q_emb = model.encode(
    dev.tweet.tolist(),
    batch_size=64,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)

_, I = index.search(q_emb.astype(np.float32), k=5)

# -------------------------------------------------------------------
# 4.  Write submission
# -------------------------------------------------------------------
run_path = ROOT / "dense_submission.tsv"
with run_path.open("w") as f:
    for pid, top5 in zip(dev.post_id, I):
        preds = [paper_ids[i] for i in top5]
        f.write(f"{pid}\t{json.dumps(preds)}\n")

print("wrote", run_path)
