# src/train_dense.py
from sentence_transformers import SentenceTransformer, InputExample, losses, SentencesDataset
from torch.utils.data import DataLoader
import json, pathlib, datetime

triples_path = pathlib.Path("/Users/agonsylejmani/Downloads/AIR/AIR#/NLP representation learning/data/triples.jsonl")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# 1. read triples ----------------------------------------------------
examples = []
with triples_path.open() as f:
    for line in f:
        j = json.loads(line)
        examples.append(InputExample(texts=[j["query"], j["pos"]]))
        examples.append(InputExample(texts=[j["query"], j["neg"]]))

# 2. wrap in Dataset & DataLoader -----------------------------------
train_dataset  = SentencesDataset(examples, model)
train_loader   = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=model.smart_batching_collate
)

# 3. define loss -----------------------------------------------------
train_loss = losses.MultipleNegativesRankingLoss(model)

# 4. fine‑tune -------------------------------------------------------
model.fit(
    train_objectives=[(train_loader, train_loss)],
    epochs=2,
    warmup_steps=int(0.1*len(train_loader)),
    output_path=f"models/dense_{datetime.datetime.now():%m%d_%H%M}"
)
