# %% [markdown]
# # Getting started
#
# ### CLEF 2025 - CheckThat! Lab  - Task 4 Scientific Web Discourse - Subtask 4b (Scientific Claim Source Retrieval)
#
# This notebook enables participants of subtask 4b to quickly get started. It includes the following:
# - Code to upload data, including:
#     - code to upload the collection set (CORD-19 academic papers' metadata)
#     - code to upload the query set (tweets with implicit references to CORD-19 papers)
# - Code to run a baseline retrieval model (BM25)
# - Code to evaluate the baseline model
#
# Participants are free to use this notebook and add their own models for the competition.
import pathlib

# %% [markdown]
# # 1) Importing data

# %%
import numpy as np
import pandas as pd

# %% [markdown]
# ## 1.a) Import the collection set
# The collection set contains metadata of CORD-19 academic papers.
#
# The preprocessed and filtered CORD-19 dataset is available on the Gitlab repository here: https://gitlab.com/checkthat_lab/clef2025-checkthat-lab/-/tree/main/task4/subtask_4b
#
# Participants should first download the file then upload it on the Google Colab session with the following steps.
#

# %%
# 1) Download the collection set from the Gitlab repository: https://gitlab.com/checkthat_lab/clef2025-checkthat-lab/-/tree/main/task4/subtask_4b
# 2) Drag and drop the downloaded file to the "Files" section (left vertical menu on Colab)
# 3) Modify the path to your local file path
DATA = pathlib.Path("/Users/agonsylejmani/Downloads/AIR/AIR#/NLP representation learning/data/")

PATH_COLLECTION_DATA = 'subtask4b_collection_data.pkl' #MODIFY PATH

# %%
df_collection = pd.read_pickle(DATA / PATH_COLLECTION_DATA)

# %%
df_collection.info()

# %%
df_collection.head()

# %% [markdown]
# ## 1.b) Import the query set
#
# The query set contains tweets with implicit references to academic papers from the collection set.
#
# The preprocessed query set is available on the Gitlab repository here: https://gitlab.com/checkthat_lab/clef2025-checkthat-lab/-/tree/main/task4/subtask_4b
#
# Participants should first download the file then upload it on the Google Colab session with the following steps.

# %%
# 1) Download the query tweets from the Gitlab repository: https://gitlab.com/checkthat_lab/clef2025-checkthat-lab/-/tree/main/task4/subtask_4b?ref_type=heads
# 2) Drag and drop the downloaded file to the "Files" section (left vertical menu on Colab)
# 3) Modify the path to your local file path
PATH_QUERY_TRAIN_DATA =DATA / 'subtask4b_query_tweets_train.tsv' #MODIFY PATH
PATH_QUERY_DEV_DATA = DATA /'subtask4b_query_tweets_dev.tsv' #MODIFY PATH

# %%
df_query_train = pd.read_csv(PATH_QUERY_TRAIN_DATA, sep = '\t')
df_query_dev = pd.read_csv(PATH_QUERY_DEV_DATA, sep = '\t')

# %%
df_query_train.head()

# %%
df_query_train.info()

# %%
df_query_dev.head()

# %%
df_query_dev.info()

# %% [markdown]
# # 2) Running the baseline
# The following code runs a BM25 baseline.
#

# %%

from rank_bm25 import BM25Okapi


# %%
# Create the BM25 corpus
corpus = df_collection[:][['title', 'abstract']].apply(lambda x: f"{x['title']} {x['abstract']}", axis=1).tolist()
cord_uids = df_collection[:]['cord_uid'].tolist()
tokenized_corpus = [doc.split(' ') for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)

# %%
def get_top_cord_uids(query):
  text2bm25top = {}
  if query in text2bm25top.keys():
      return text2bm25top[query]
  else:
      tokenized_query = query.split(' ')
      doc_scores = bm25.get_scores(tokenized_query)
      indices = np.argsort(-doc_scores)[:5]
      bm25_topk = [cord_uids[x] for x in indices]

      text2bm25top[query] = bm25_topk
      return bm25_topk


# %%
# Retrieve topk candidates using the BM25 model
df_query_train['bm25_topk'] = df_query_train['tweet_text'].apply(lambda x: get_top_cord_uids(x))
df_query_dev['bm25_topk'] = df_query_dev['tweet_text'].apply(lambda x: get_top_cord_uids(x))

# %% [markdown]
# # 3) Evaluating the baseline
# The following code evaluates the BM25 retrieval baseline on the query set using the Mean Reciprocal Rank score (MRR@5).

# %%
# Evaluate retrieved candidates using MRR@k
def get_performance_mrr(data, col_gold, col_pred, list_k = [1, 5, 10]):
    d_performance = {}
    for k in list_k:
        data["in_topx"] = data.apply(lambda x: (1/([i for i in x[col_pred][:k]].index(x[col_gold]) + 1) if x[col_gold] in [i for i in x[col_pred][:k]] else 0), axis=1)
        #performances.append(data["in_topx"].mean())
        d_performance[k] = data["in_topx"].mean()
    return d_performance


# %%
results_train = get_performance_mrr(df_query_train, 'cord_uid', 'bm25_topk')
results_dev = get_performance_mrr(df_query_dev, 'cord_uid', 'bm25_topk')
# Printed MRR@k results in the following format: {k: MRR@k}
print(f"Results on the train set: {results_train}")
print(f"Results on the dev set: {results_dev}")


# %% [markdown]
# ## 3 bis) Evaluating our dense‑retrieval run file
# -------------------------------------------------
# This cell loads the TSV produced by `run_retrieve.py`, turns the JSON strings
# into lists, joins them with the dev‑set ground‑truth, and prints MRR@k.

# %%
import json, pathlib

# --- 1. Path to the run file --------------------------------------------------
RUN_FILE = pathlib.Path("/Users/agonsylejmani/Downloads/AIR/AIR#/NLP representation learning/dense_submission.tsv")

# --- 2. Load the TSV (no header row) ------------------------------------------
df_run = pd.read_csv(
    RUN_FILE,
    sep='\t',
    names=['post_id', 'preds_json'],
    dtype={'post_id': str, 'preds_json': str}   # force string right away
)

df_run['preds'] = df_run['preds_json'].apply(json.loads)

# --- make sure both sides are strings -----------------------------------------
df_query_dev['post_id'] = df_query_dev['post_id'].astype(str)

# --- merge & evaluate ---------------------------------------------------------
df_eval = df_query_dev.merge(df_run[['post_id', 'preds']], on='post_id')
results_dense = get_performance_mrr(df_eval, 'cord_uid', 'preds')
print(f"Dense model results on the dev set (MRR@k): {results_dense}")


# %% [markdown]
# ### (Optional) Replace the BM25 column so the rest of the notebook uses the dense results
# Uncomment the next two lines if you want `df_query_dev['bm25_topk']` to hold the
# neural rankings instead of the keyword baseline.

# %%
# df_query_dev['bm25_topk'] = df_run.set_index('post_id').loc[df_query_dev.post_id, 'preds'].values
# print("bm25_topk column now contains dense predictions.")



# %% [markdown]
# # 4) Exporting results to prepare the submission on Codalab

# %%
df_query_dev['preds'] = df_query_dev['bm25_topk'].apply(lambda x: x[:5])

# %%
df_query_dev[['post_id', 'preds']].to_csv('predictions.tsv', index=None, sep='\t')

# %%



