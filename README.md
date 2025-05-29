# AIR project - CheckThat!
## Neural Re-ranking Model

### 📦 Requirements

Install the dependencies using `pip`. You can use a virtual environment for isolation.

```bash
pip install torch transformers pandas scikit-learn tqdm nltk
```


### ⚙️ Configuration
Edit the config.json file to set paths, model names, batch size, max length, etc. Example:
<pre><code>json { "model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2", "batch_size": 32, "learning_rate": 3e-05, "total_epochs": 5, "accum_steps": 2, "max_length": 512, "bm25_top_k": 1000, "log_dir": "runs/re_rank_experiment", "save_dir": "rerank_model/Models/Model1", "collection_path": "subtask4b_collection_data.pkl", "train_query_path": "subtask4b_query_tweets_train.tsv", "test_query_path": "subtask4b_query_tweets_test_gold.tsv", "margin_loss": 1.0, "weight_decay": 0.01, "scheduler": "linear", "seed": 42 } </code></pre>

### 🚀 Running the Code
1. Training
Train the re-ranking model with:
```bash
python train.py --config config.json
```
The models for each epoch, as well as the log files and metrics will be saved at "save_dir" location set in config.

3. Evaluation
Evaluate the trained model using:
```bash
python evaluate.py --config config.json
```
Evaluation metrics such as MRR@1, MRR@5, and MRR@10 will be logged to eval_log.txt and metrics.json at "save_dir" location set in config. The predictions will be save as .tsv files at "{save_dir}/predictions".

### Configuration used for our best model
<pre><code>json { "model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2", "batch_size": 32, "learning_rate": 3e-05, "total_epochs": 5, "accum_steps": 2, "max_length": 512, "bm25_top_k": 1000, "log_dir": "runs/re_rank_experiment", "save_dir": "rerank_model/Models/Model1", "collection_path": "subtask4b_collection_data.pkl", "train_query_path": "subtask4b_query_tweets_train.tsv", "test_query_path": "subtask4b_query_tweets_test_gold.tsv", "margin_loss": 1.0, "weight_decay": 0.01, "scheduler": "linear", "seed": 42 } </code></pre>
