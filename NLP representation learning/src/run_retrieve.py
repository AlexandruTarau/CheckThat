# src/run_retrieve.py
import pandas as pd, faiss, pickle, pathlib
from sentence_transformers import SentenceTransformer
import numpy as np, json, tqdm, ast

DATA = pathlib.Path("data")
model = SentenceTransformer(sorted(pathlib.Path("models").glob("dense_*"))[-1])
index = faiss.read_index("models/paper_index.faiss")
paper_ids = pd.read_pickle("models/paper_ids.pkl")["cord_uid"].tolist()

dev = pd.read_csv(DATA/"subtask4b_query_tweets_dev.tsv",
                  sep="\t", names=["post_id","tweet","label"])
q_emb = model.encode(dev.tweet.tolist(), batch_size=64,
                     show_progress_bar=True, convert_to_numpy=True,
                     normalize_embeddings=True)

D, I = index.search(q_emb.astype(np.float32), 5)

with open("dense_submission.tsv","w") as f:
    for pid, top5 in zip(dev.post_id, I):
        preds = [paper_ids[i] for i in top5]
        f.write(f"{pid}\t{json.dumps(preds)}\n")
print("wrote dense_submission.tsv")
