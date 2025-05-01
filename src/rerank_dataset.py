from torch.utils.data import Dataset
from transformers import BertTokenizer

# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


# Create a custom dataset class
class ReRankDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        query = self.data.iloc[idx]['query']
        doc = self.data.iloc[idx]['doc']
        label = self.data.iloc[idx]['label']

        # Tokenize the query-doc pair
        inputs = tokenizer(query, doc, padding='max_length', truncation=True, max_length=512, return_tensors="pt")

        # Flatten the tensors (needed for BERT)
        item = {key: val.squeeze(0) for key, val in inputs.items()}
        item['labels'] = label  # Add the label (0 or 1)

        return item
