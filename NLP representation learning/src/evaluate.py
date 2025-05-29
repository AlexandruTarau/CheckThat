import ast, pathlib, pandas as pd, json

ROOT = pathlib.Path(__file__).parent.parent
GOLD = ROOT / "data" / "subtask4b_query_tweets_test_gold.tsv"
RUN  = ROOT / "output" / "test_predictions.tsv"

# ------------------------------------------------------------------
# 1. gold labels → dict(str_id → gold paper_id)
# ------------------------------------------------------------------
gold_df = pd.read_csv(
    GOLD,
    sep="\t",
    names=["post_id", "tweet_text", "cord_uid"],
    dtype={"post_id": str}
)
gold = dict(zip(gold_df.post_id, gold_df.cord_uid))

# ------------------------------------------------------------------
# 2. run file → dict(str_id → [paper_id1, paper_id2, …])
#    (supports both long and wide formats)
# ------------------------------------------------------------------
def load_run(path):
    df = pd.read_csv(path, sep="\t", dtype={"tweet_id": str, "post_id": str})

    # long format: tweet_id | paper_id | rank | score
    if {"tweet_id", "paper_id", "rank"}.issubset(df.columns):
        run = (df.sort_values(["tweet_id", "rank"])
                 .groupby("tweet_id")["paper_id"].apply(list)
                 .to_dict())
        max_k = int(df["rank"].max())
    # wide format: post_id | preds (JSON list)
    else:
        run = (df.set_index("post_id")["preds"]
                 .apply(ast.literal_eval)
                 .to_dict())
        max_k = len(next(iter(run.values())))
    return run, max_k

run, max_k = load_run(RUN)

# ------------------------------------------------------------------
# 3. metric helpers
# ------------------------------------------------------------------
def mrr_at_k(run_dict, gold_dict, k):
    rr = []
    for qid, cand in run_dict.items():
        gold_pid = gold_dict.get(qid)
        if gold_pid is None:
            continue
        try:
            rank = cand[:k].index(gold_pid) + 1   # consider top-k slice
            rr.append(1 / rank)
        except ValueError:
            rr.append(0.0)
    return sum(rr) / len(rr) if rr else 0.0

# compute requested cut-offs
for k in (1, 5, 10):
    if k <= max_k:
        mrr = mrr_at_k(run, gold, k)
        print(f"MRR@{k}: {mrr:.4f}")
    else:
        print(f"MRR@{k}: n/a  (run file only has {max_k} predictions per query)")
