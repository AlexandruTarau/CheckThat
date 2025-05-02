# src/build_index.py
import pandas as pd, faiss, pathlib, numpy as np
from sentence_transformers import SentenceTransformer

DATA = pathlib.Path("data")
model = SentenceTransformer(sorted(pathlib.Path("models").glob("dense_*"))[-1])
papers = pd.read_pickle(DATA/"subtask4b_collection_data.pkl")
texts = [f"{r.title}. {r.abstract}" for r in papers.itertuples()]
emb = model.encode(texts, batch_size=128, show_progress_bar=True,
                   convert_to_numpy=True, normalize_embeddings=True)

index = faiss.IndexFlatIP(emb.shape[1])
index.add(emb.astype(np.float32))
faiss.write_index(index, "models/paper_index.faiss")
papers[["cord_uid"]].to_pickle("models/paper_ids.pkl")
print("indexed", len(texts), "papers")
