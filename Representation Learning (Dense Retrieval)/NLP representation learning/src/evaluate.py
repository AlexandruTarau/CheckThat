import ast, pathlib, pandas as pd, json

ROOT = pathlib.Path(__file__).parent.parent
GOLD = ROOT / "data" / "subtask4b_query_tweets_test_gold.tsv"
RUN  = ROOT / "output" / "test_predictions.tsv"

# load gold labels into dict(post_id -> paper_id)
gold_df = pd.read_csv(
    GOLD,
    sep="\t",
    names=["post_id", "tweet_text", "cord_uid"],
    dtype={"post_id": str}
)
gold = dict(zip(gold_df.post_id, gold_df.cord_uid))

# parse run file into dict(post_id -> list of paper ids)
def load_run(path):
    df = pd.read_csv(path, sep="\t", dtype={"tweet_id": str, "post_id": str})

    # detect long format
    if {"tweet_id", "paper_id", "rank"}.issubset(df.columns):
        run = (
            df.sort_values(["tweet_id", "rank"]
            )
            .groupby("tweet_id")["paper_id"].apply(list)
            .to_dict()
        )
        max_k = int(df["rank"].max())
    # handle wide format
    else:
        run = (
            df.set_index("post_id")["preds"]
            .apply(ast.literal_eval)
            .to_dict()
        )
        max_k = len(next(iter(run.values())))
    return run, max_k

run, max_k = load_run(RUN)

# define mrr calculation helper
def mrr_at_k(run_dict, gold_dict, k):
    rr = []
    for qid, cand in run_dict.items():
        gold_pid = gold_dict.get(qid)
        if gold_pid is None:
            continue
        try:
            rank = cand[:k].index(gold_pid) + 1
            rr.append(1 / rank)
        except ValueError:
            rr.append(0.0)
    return sum(rr) / len(rr) if rr else 0.0

# compute mrr for k in [1,5,10]
for k in (1, 5, 10):
    if k <= max_k:
        mrr = mrr_at_k(run, gold, k)
        print(f"mrr@{k}: {mrr:.4f}")
    else:
        print(f"mrr@{k}: n/a  (only {max_k} preds per query)")
