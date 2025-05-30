# src/predict_test.py   ––  Hybrid BM25 + dense  (macOS-safe)

# ------------------------------------------------------------------
# 0.  MUST come before importing faiss / numpy heavy libs
# ------------------------------------------------------------------
import os
os.environ["OMP_NUM_THREADS"]      = "1"      # ← prevent OpenMP crashes
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"   # ← macOS / Intel quirk

# ------------------------------------------------------------------
# 1.  Imports
# ------------------------------------------------------------------
import pathlib, pickle, faiss, pandas as pd, numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

faiss.omp_set_num_threads(1)                    # ← belt-and-braces

# ------------------------------------------------------------------
# 2.  Paths
# ------------------------------------------------------------------
root_dir          = pathlib.Path(__file__).parent.parent
test_path         = root_dir / "data"   / "subtask4b_query_tweets_test.tsv"
collection_path   = root_dir / "data"   / "subtask4b_collection_data.pkl"
paper_ids_path    = root_dir / "models" / "paper_ids.pkl"
faiss_index_path  = root_dir / "models" / "paper_index.faiss"
model_dir         = root_dir / "models"

# ------------------------------------------------------------------
# 3.  Load encoder checkpoint
# ------------------------------------------------------------------
model_path = sorted(model_dir.glob("dense_*"), reverse=True)[0]
model      = SentenceTransformer(str(model_path))

# ------------------------------------------------------------------
# 4.  Data
# ------------------------------------------------------------------
test_df = pd.read_csv(test_path, sep="\t", dtype={"post_id": str})

with open(collection_path, "rb") as f:
    col_obj = pickle.load(f)

# either dict or DataFrame → make DataFrame with index = cord_uid
if isinstance(col_obj, pd.DataFrame):
    collection_df = col_obj.set_index("cord_uid")
else:
    collection_df = pd.DataFrame.from_dict(col_obj, orient="index")
    collection_df.index.name = "cord_uid"

for col in ("title", "abstract"):              # guard missing cols
    if col not in collection_df.columns:
        collection_df[col] = ""

with open(paper_ids_path, "rb") as f:
    tmp = pickle.load(f)
    if isinstance(tmp, pd.DataFrame) and tmp.shape[1] == 1:
        paper_ids = tmp.iloc[:, 0].astype(str).tolist()
    else:
        paper_ids = [str(x) for x in list(tmp)]

index = faiss.read_index(str(faiss_index_path))

# ------------------------------------------------------------------
# 5.  Build BM25 (aligned with paper_ids order)
# ------------------------------------------------------------------
def safe_text(row):
    return (str(row["title"]) + " " + str(row["abstract"])).lower()

paper_texts = [safe_text(collection_df.loc[pid])
               if pid in collection_df.index else ""
               for pid in paper_ids]

bm25 = BM25Okapi([doc.split() for doc in paper_texts])

# ------------------------------------------------------------------
# 6.  Retrieval loop  (hybrid score)
# ------------------------------------------------------------------
TOP_K_DENSE = 10      # dense shortlist
TOP_OUT     = 10      # final list
ALPHA       = 0.4     # BM25 weight

results = []

for _, row in test_df.iterrows():
    qid, tweet = row["post_id"], row["tweet_text"]

    # dense embedding  ––  make it contiguous float32 row
    q_vec = model.encode([tweet], convert_to_numpy=True)
    q_vec = np.ascontiguousarray(q_vec.astype("float32"))   # ← crucial

    D, I = index.search(q_vec, TOP_K_DENSE)                 # cosine scores
    bm25_scores_all = bm25.get_scores(tweet.lower().split())

    hits = []
    for dense_score, idx in zip(D[0], I[0]):
        pid          = paper_ids[idx]
        hybrid       = ALPHA * bm25_scores_all[idx] + (1 - ALPHA) * dense_score
        hits.append((hybrid, pid))

    hits.sort(reverse=True)
    for rank, (score, pid) in enumerate(hits[:TOP_OUT], 1):
        results.append([qid, pid, rank, float(score)])

# ------------------------------------------------------------------
# 7.  Save run file (long format)
# ------------------------------------------------------------------
out_dir  = root_dir / "output"
out_dir.mkdir(exist_ok=True)
out_path = out_dir / "test_predictions.tsv"

pd.DataFrame(results,
             columns=["tweet_id", "paper_id", "rank", "score"]
            ).to_csv(out_path, sep="\t", index=False)

print(f"✅  Predictions saved to {out_path}")
