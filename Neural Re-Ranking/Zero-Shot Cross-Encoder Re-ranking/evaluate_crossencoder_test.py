import pandas as pd
import ast

# === Load predictions and gold labels ===
df_preds = pd.read_csv("submission_bm25_test_gold.tsv", sep="\t")
df_gold = pd.read_csv("subtask4b_query_tweets_test_gold.tsv", sep="\t")

# === Merge on post_id ===
df = df_preds.merge(df_gold[['post_id', 'cord_uid']], on='post_id')

# === Convert prediction strings to lists ===
df['preds'] = df['preds'].apply(ast.literal_eval)

# === Evaluate MRR ===
def get_mrr(df, k):
    reciprocal_ranks = []
    for _, row in df.iterrows():
        gold = row['cord_uid']
        preds = row['preds'][:k]
        if gold in preds:
            rank = preds.index(gold) + 1
            reciprocal_ranks.append(1 / rank)
        else:
            reciprocal_ranks.append(0)
    return sum(reciprocal_ranks) / len(reciprocal_ranks)

# === Print results ===
for k in [1, 5, 10]:
    score = get_mrr(df, k)
    print(f"MRR@{k}: {score:.4f}")
