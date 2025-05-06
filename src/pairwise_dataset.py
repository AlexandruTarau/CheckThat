# AIR/src/pairwise_dataset.py
from torch.utils.data import Dataset
import torch

class PairwiseDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=256):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        query = row['query']
        rel_doc = row['rel_doc']
        non_rel_doc = row['non_rel_doc']

        rel = self.tokenizer(query, rel_doc, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
        non_rel = self.tokenizer(query, non_rel_doc, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')

        return {
            'rel_input_ids': rel['input_ids'].squeeze(0),
            'rel_attention_mask': rel['attention_mask'].squeeze(0),
            'non_rel_input_ids': non_rel['input_ids'].squeeze(0),
            'non_rel_attention_mask': non_rel['attention_mask'].squeeze(0),
        }
