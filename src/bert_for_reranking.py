import torch.nn as nn
from transformers import BertModel


class BertForReRanking(nn.Module):
    def __init__(self):
        super(BertForReRanking, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.scorer = nn.Linear(self.bert.config.hidden_size, 1)
        # self.scorer = nn.Sequential(
        #     nn.Linear(self.bert.config.hidden_size, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 1)
        # )

    def forward(self, input_ids, attention_mask):
        # Get [CLS] embedding
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # shape: (batch_size, hidden_size)
        score = self.scorer(cls_output).squeeze(-1)      # shape: (batch_size,)
        return score
