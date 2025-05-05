# src/make_triples.py
import pandas as pd, random, json, pathlib
from tqdm import tqdm

DATA = pathlib.Path("/Users/agonsylejmani/Downloads/AIR/AIR#/NLP representation learning/data/")
OUT  = pathlib.Path("/Users/agonsylejmani/Downloads/AIR/AIR#/NLP representation learning/data/triples.jsonl")
N_HARD = 5          # BM25 negatives per query
N_RAND = 2          # random negatives

# 1. load tweets & papers
train = pd.read_csv(DATA / "subtask4b_query_tweets_train.tsv",
                    sep="\t", names=["post_id","tweet","label"])
papers = pd.read_pickle(DATA / "subtask4b_collection_data.pkl")
pid2text = {r.cord_uid: f"{r.title}. {r.abstract}" for r in papers.itertuples()}

# 2. pre‑compute BM25 ranks (already in notebook output)
bm25 = pd.read_csv(DATA / "bm25_train_ranking.tsv", sep="\t",
                   names=["post_id","candidates"])   # save this from notebook

# 3. build triples
with OUT.open("w") as f:
    for row in tqdm(train.itertuples(), total=len(train)):
        q = row.tweet
        pos_id = row.label
        hard = json.loads(bm25.loc[bm25.post_id==row.post_id,"candidates"].item())
        hard_negs = [d for d in hard if d!=pos_id][:N_HARD]
        rand_negs = random.sample(
            [cid for cid in pid2text if cid not in hard_negs+[pos_id]],
            k=N_RAND)
        for neg_id in hard_negs+rand_negs:
            triple = {"query":q,
                      "pos": pid2text[pos_id],
                      "neg": pid2text[neg_id]}
            f.write(json.dumps(triple)+"\n")
print("saved", OUT)
