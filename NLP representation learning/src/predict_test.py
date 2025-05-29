import pandas as pd
import pickle
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import pathlib

# Paths
root_dir = pathlib.Path(__file__).parent.parent
test_path = root_dir / "data" / "subtask4b_query_tweets_test.tsv"
collection_path = root_dir / "data" / "subtask4b_collection_data.pkl"
paper_ids_path = root_dir / "models" / "paper_ids.pkl"
faiss_index_path = root_dir / "models" / "paper_index.faiss"
model_dir = root_dir / "models"

# Get latest dense model
model_dirs = sorted(model_dir.glob("dense_*"), reverse=True)
assert model_dirs, "No trained models found. Train first!"
model_path = model_dirs[0]

# Load model
model = SentenceTransformer(str(model_path))

# Load test queries
test_df = pd.read_csv(test_path, sep='\t')

# Load collection (id to metadata/text)
with open(collection_path, "rb") as f:
    collection = pickle.load(f)  # Usually a dict: paper_id -> {"title":..., "abstract":...}

# Load paper_ids and ensure it's a list
with open(paper_ids_path, "rb") as f:
    paper_ids = pickle.load(f)
    # Robustly convert to list if needed
    if isinstance(paper_ids, pd.DataFrame) and paper_ids.shape[1] == 1:
        paper_ids = paper_ids.iloc[:, 0].tolist()
    elif hasattr(paper_ids, "tolist"):
        paper_ids = paper_ids.tolist()
    # Now paper_ids should be a plain list
    assert isinstance(paper_ids, list), f"paper_ids is not a list, but {type(paper_ids)}"


# Load paper_ids and ensure it's a list
with open(paper_ids_path, "rb") as f:
    paper_ids = pickle.load(f)
    if isinstance(paper_ids, pd.DataFrame) and paper_ids.shape[1] == 1:
        paper_ids = paper_ids.iloc[:, 0].tolist()
    elif hasattr(paper_ids, "tolist"):
        paper_ids = paper_ids.tolist()
    assert isinstance(paper_ids, list), f"paper_ids is not a list, but {type(paper_ids)}"

# Load FAISS index
index = faiss.read_index(str(faiss_index_path))

# --- ADD THIS BLOCK RIGHT HERE ---
print("Number of paper_ids:", len(paper_ids))
print("FAISS index ntotal:", index.ntotal)
print("FAISS index dimension d:", index.d)
# Test the embedding dimension
sample_vec = model.encode(["test"], convert_to_numpy=True)
print("Sample embedding shape:", sample_vec.shape)
assert sample_vec.shape[1] == index.d, "Embedding dimension does not match FAISS index dimension!"
assert len(paper_ids) == index.ntotal, "paper_ids and FAISS index size do not match!"
# --- END DEBUG BLOCK ---


# Load FAISS index
index = faiss.read_index(str(faiss_index_path))

results = []
top_k = 5

for idx, row in test_df.iterrows():
    tweet_id = row['post_id']  # use the correct column name!
    tweet_text = row['tweet_text']

    query_vec = model.encode([tweet_text], convert_to_numpy=True).astype("float32")
    D, I = index.search(query_vec, top_k)

    for rank, (idx_in_index, score) in enumerate(zip(I[0], D[0]), 1):
        paper_id = paper_ids[idx_in_index]
        results.append([tweet_id, paper_id, rank, float(score)])

# Save all predictions as before
out_dir = root_dir / "output"
out_dir.mkdir(exist_ok=True)
out_path = out_dir / "test_predictions.tsv"
pd.DataFrame(results, columns=['tweet_id', 'paper_id', 'rank', 'score']).to_csv(out_path, sep='\t', index=False)
print(f"Predictions saved to {out_path}")
