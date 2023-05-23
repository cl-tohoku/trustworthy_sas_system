import torch
import sys

sys.path.append("..")
from model.base import ModelBase, Attention


class LSTM(ModelBase):
    def __init__(self, config, vectors):
        super().__init__(config)
        self.embedding = torch.nn.Embedding.from_pretrained(vectors.vectors.float(), freeze=False)
        self.layer_size = 3
        self.lstm = torch.nn.LSTM(300, self.config.hidden_size // 2, num_layers=self.layer_size,
                                  bidirectional=True, dropout=self.config.dropout_rate)

        for i in range(self.config.output_size):
            setattr(self, "attn_{}".format(i + 1), Attention(self.config, self.config.hidden_size))

    def forward(self, input_ids, attention=False, inputs_embeds=False, sent_vector=False):
        # embedding
        if not inputs_embeds:
            output = self.embedding(input_ids)
        else:
            output = input_ids

        h_0 = torch.zeros(self.layer_size * 2, output.shape[1], self.config.hidden_size // 2).cuda()
        c_0 = torch.zeros(self.layer_size * 2, output.shape[1], self.config.hidden_size // 2).cuda()
        self.lstm.flatten_parameters()
        output, (hidden, cell) = self.lstm(output, (h_0, c_0))

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
        elif sent_vector:
            return output, query_tensor
        else:
            return output
