import torch
import sys

sys.path.append("..")
from model.base import ModelBase, Attention


class FFNN(ModelBase):
    def __init__(self, config, vectors):
        super().__init__(config)
        self.embedding = torch.nn.Embedding.from_pretrained(vectors.vectors.float(), freeze=True)
        self.fc_1 = torch.nn.Linear(vectors.vectors.shape[1], self.config.hidden_size)
        self.fc_2 = torch.nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.fc_3 = torch.nn.Linear(self.config.hidden_size, self.config.hidden_size)

        for i in range(self.config.output_size):
            setattr(self, "attn_{}".format(i + 1), Attention(self.config, self.config.hidden_size))

    def forward(self, input_ids, attention=False):
        # embedding
        output = self.embedding(input_ids)

        for i in range(3):
            output = getattr(self, "fc_{}".format(i + 1))(output)
            output = self.dropout(output)
            output = self.relu(output)

        # attention
        query_tensor = torch.mean(output, dim=1, keepdim=True)
        outputs, attention_weights = [], []
        for i in range(self.config.output_size):
            output_i, attention_w = getattr(self, "attn_{}".format(i + 1))(query_tensor, output)
            outputs.append(output_i)
            attention_weights.append(attention_w)

        # concat
        output = torch.cat(outputs, dim=1)

        if attention:
            return output, attention_weights
        else:
            return output
