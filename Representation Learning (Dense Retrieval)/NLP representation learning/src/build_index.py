import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path

# set project paths
tree_root  = Path("/Users/agonsylejmani/Downloads/AIR/AIR#/NLP representation learning")
data_dir   = tree_root / "data"
models_dir = tree_root / "models"

# load latest transformer checkpoint
model_dir = sorted(models_dir.glob("dense_*"))[-1]
model     = SentenceTransformer(str(model_dir))

# read collection data and prepare texts
papers = pd.read_pickle(data_dir / "subtask4b_collection_data.pkl")
texts  = [f"{r.title}. {r.abstract}" for r in papers.itertuples()]

# encode texts into embeddings
embeddings = model.encode(
    texts,
    batch_size=128,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)

# build inner-product index
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings.astype(np.float32))

# save index and paper ids
faiss.write_index(index, str(models_dir / "paper_index.faiss"))
papers[["cord_uid"]].to_pickle(str(models_dir / "paper_ids.pkl"))

print(f"indexed {len(texts)} papers → {models_dir/'paper_index.faiss'}")
