# src/train_dense.py
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import json, pathlib, tqdm, datetime, os

triples_path = pathlib.Path("data/triples.jsonl")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

examples = []
with triples_path.open() as f:
    for line in f:
        j = json.loads(line)
        examples.append(
            InputExample(texts=[j["query"], j["pos"]]))
        examples.append(
            InputExample(texts=[j["query"], j["neg"]]))

loader = DataLoader(examples, batch_size=32, shuffle=True, drop_last=True)
loss = losses.MultipleNegativesRankingLoss(model)

model.fit(loader, epochs=2, warmup_steps=1000,
          output_path="models/dense_"+datetime.datetime.now().strftime("%m%d_%H%M"))
