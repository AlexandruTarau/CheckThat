import pandas as pd, random, json, pathlib
from tqdm import tqdm

DATA = pathlib.Path("/Users/agonsylejmani/Downloads/AIR/AIR#/NLP representation learning/data/")
OUT  = pathlib.Path("/Users/agonsylejmani/Downloads/AIR/AIR#/NLP representation learning/data/triples.jsonl")
N_HARD = 5          # BM25 negatives per query
N_RAND = 2          # random negatives

# 1. load tweets & papers
train = (
    pd.read_csv(DATA / "subtask4b_query_tweets_train.tsv", sep="\t")
      .rename(columns={"tweet_text": "tweet", "cord_uid": "label"})
)

papers = pd.read_pickle(DATA / "subtask4b_collection_data.pkl")
pid2text = {r.cord_uid: f"{r.title}. {r.abstract}" for r in papers.itertuples()}

# 2. load BM25 predictions
bm25 = (
    pd.read_csv(DATA / "predictions.tsv", sep="\t", names=["post_id", "preds"])
      .assign(preds=lambda df: df.preds.fillna("[]"))        # guard against NaNs
)

# 3. build triples
with OUT.open("w") as f:
    for row in tqdm(train.itertuples(index=False), total=len(train)):
        q       = row.tweet
        pos_id  = row.label

        # fetch & parse hard negatives
        preds_str = bm25.loc[bm25.post_id == row.post_id, "preds"]
        preds_str = preds_str.item() if len(preds_str) else "[]"
        try:
            hard = json.loads(preds_str)
        except json.JSONDecodeError:
            hard = []

        hard_negs = [cid for cid in hard if cid != pos_id][:N_HARD]
        rand_negs = random.sample(
            [cid for cid in pid2text if cid not in hard_negs + [pos_id]],
            k=N_RAND
        )

        for neg_id in hard_negs + rand_negs:
            triple = {
                "query": q,
                "pos":   pid2text[pos_id],
                "neg":   pid2text[neg_id],
            }
            f.write(json.dumps(triple) + "\n")

print("saved", OUT)
