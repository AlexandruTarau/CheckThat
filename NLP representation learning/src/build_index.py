# src/build_index.py
"""
Create a FAISS index of all papers so that the dense‑retrieval model
can find nearest neighbours quickly.

• Expects:
    project_root/
    ├── data/subtask4b_collection_data.pkl
    └── models/dense_0505_1245/   (your trained checkpoint)

• Produces:
    models/paper_index.faiss
    models/paper_ids.pkl
"""

import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path

# -------------------------------------------------------------------
# 1.  Paths (edit ROOT only if your repo lives somewhere else)
# -------------------------------------------------------------------
ROOT   = Path("/Users/agonsylejmani/Downloads/AIR/AIR#/NLP representation learning")
DATA   = ROOT / "data"
MODELS = ROOT / "models"

# -------------------------------------------------------------------
# 2.  Load the Sentence‑Transformers checkpoint
# -------------------------------------------------------------------
model_dir = MODELS / "dense_0505_1245"        # ← or use glob("dense_*")[-1] for latest
model = SentenceTransformer(str(model_dir))   # cast Path → str

# -------------------------------------------------------------------
# 3.  Read paper collection and encode title + abstract
# -------------------------------------------------------------------
papers = pd.read_pickle(DATA / "subtask4b_collection_data.pkl")
texts  = [f"{r.title}. {r.abstract}" for r in papers.itertuples()]

emb = model.encode(
    texts,
    batch_size=128,                 # lower if GPU/CPU memory is small
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True       # cosine → dot‑product
)

# -------------------------------------------------------------------
# 4.  Build & save FAISS index
# -------------------------------------------------------------------
index = faiss.IndexFlatIP(emb.shape[1])       # inner‑product (cosine when normalised)
index.add(emb.astype(np.float32))

faiss.write_index(index, str(MODELS / "paper_index.faiss"))
papers[["cord_uid"]].to_pickle(str(MODELS / "paper_ids.pkl"))

print(f"indexed {len(texts)} papers → {MODELS/'paper_index.faiss'}")
