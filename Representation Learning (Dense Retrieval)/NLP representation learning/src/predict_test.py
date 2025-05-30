#!/usr/bin/env python3
import os
os.environ["OMP_NUM_THREADS"]      = "1"      
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"   

import pathlib, pickle, faiss, pandas as pd, numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# restrict faiss threads
faiss.omp_set_num_threads(1)

# define paths
root_dir         = pathlib.Path(__file__).parent.parent
test_path        = root_dir / "data" / "subtask4b_query_tweets_test.tsv"
collection_path  = root_dir / "data" / "subtask4b_collection_data.pkl"
paper_ids_path   = root_dir / "models" / "paper_ids.pkl"
faiss_index_path = root_dir / "models" / "paper_index.faiss"
model_dir        = root_dir / "models"

# load encoder checkpoint
model_path = sorted(model_dir.glob("dense_*"), reverse=True)[0]
model      = SentenceTransformer(str(model_path))

# load test data
load_df = pd.read_csv(test_path, sep="\t", dtype={"post_id": str})

# load collection data
with open(collection_path, "rb") as f:
    col_obj = pickle.load(f)
if isinstance(col_obj, pd.DataFrame):
    collection_df = col_obj.set_index("cord_uid")
else:
    collection_df = pd.DataFrame.from_dict(col_obj, orient="index")
    collection_df.index.name = "cord_uid"

# ensure title and abstract exist
for col in ("title", "abstract"):
    if col not in collection_df.columns:
        collection_df[col] = ""

# load paper ids
with open(paper_ids_path, "rb") as f:
    tmp = pickle.load(f)
    if isinstance(tmp, pd.DataFrame) and tmp.shape[1] == 1:
        paper_ids = tmp.iloc[:, 0].astype(str).tolist()
    else:
        paper_ids = [str(x) for x in tmp]

# load faiss index
index = faiss.read_index(str(faiss_index_path))

# prepare bm25 corpus (ordered by paper_ids)
def _safe_text(row):
    return (str(row["title"]) + " " + str(row["abstract"])).lower()
paper_texts = [ _safe_text(collection_df.loc[pid]) if pid in collection_df.index else "" for pid in paper_ids ]
bm25 = BM25Okapi([doc.split() for doc in paper_texts])

# retrieval settings
TOP_K_DENSE = 10   
TOP_OUT     = 10   
ALPHA       = 0.4  

results = []
for _, row in load_df.iterrows():
    qid, tweet = row["post_id"], row["tweet_text"]
    # compute dense vector
    q_vec = model.encode([tweet], convert_to_numpy=True)
    q_vec = np.ascontiguousarray(q_vec.astype("float32"))

    # dense search
    D, I = index.search(q_vec, TOP_K_DENSE)
    bm25_scores = bm25.get_scores(tweet.lower().split())

    # combine scores
    candidates = []
    for dense_score, idx in zip(D[0], I[0]):
        pid    = paper_ids[idx]
        hybrid = ALPHA * bm25_scores[idx] + (1 - ALPHA) * dense_score
        candidates.append((hybrid, pid))
    candidates.sort(reverse=True)

    # record top results
    for rank, (score, pid) in enumerate(candidates[:TOP_OUT], 1):
        results.append([qid, pid, rank, float(score)])

# save predictions
out_dir  = root_dir / "output"
out_dir.mkdir(exist_ok=True)
out_path = out_dir / "test_predictions.tsv"

pd.DataFrame(results, columns=["tweet_id","paper_id","rank","score"]).to_csv(out_path, sep="\t", index=False)

print(f" predictions saved to {out_path}")
