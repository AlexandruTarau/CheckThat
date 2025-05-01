import torch.nn as nn
from transformers import BertModel

class BertForReRanking(nn.Module):
    def __init__(self):
        super(BertForReRanking, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.classifier = nn.Linear(self.bert.config.hidden_size, 2)  # Binary classification (relevant or not)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # Use [CLS] token
        logits = self.classifier(cls_output)
        return logits
