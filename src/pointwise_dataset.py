import torch

class PointwiseDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, max_length=512):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        query = self.df.iloc[idx]['query']
        doc = self.df.iloc[idx]['doc']
        label = self.df.iloc[idx]['label']

        # Encoding the query and document pairs for the model
        encoding = self.tokenizer(query, doc,
                                  padding='max_length',  # Pad all sequences to max_length
                                  truncation=True,        # Truncate sequences to max_length
                                  max_length=self.max_length,
                                  return_tensors='pt')

        return {
            'input_ids': encoding['input_ids'].squeeze(),  # [batch_size, seq_len]
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(label, dtype=torch.long)
        }
