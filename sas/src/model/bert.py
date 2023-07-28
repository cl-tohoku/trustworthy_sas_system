import sys
import torch
from transformers import BertModel

sys.path.append("..")
from model.base import ModelBase, Attention


class SimpleBert(ModelBase):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel.from_pretrained("cl-tohoku/bert-base-japanese-char-v2")
        self.dropout = torch.nn.Dropout(config.dropout_rate)
        self.output_size = config.output_size

        for i in range(self.output_size):
            setattr(self, "attn_{}".format(i + 1), Attention(self.config, 768))

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, attention=False,
                inputs_embeds=False, sent_vector=False, batch_sum=False):

        if inputs_embeds:
            bert_output = self.bert(inputs_embeds=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        else:
            bert_output = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        # extract last & make query
        last_hidden_state = bert_output["last_hidden_state"]
        query_tensor = last_hidden_state[:, 0, :].unsqueeze(1)

        # parallel
        outputs, attention_weights = [], []
        for i in range(self.output_size):
            output, attention_w = getattr(self, "attn_{}".format(i + 1))(query_tensor, last_hidden_state)
            outputs.append(output)
            attention_weights.append(attention_w)

        # concat
        output = torch.cat(outputs, dim=1)
        attention_weights = torch.stack(attention_weights).permute(1, 0, 2)

        if attention:
            return output, attention_weights
        elif sent_vector:
            return output, query_tensor
        elif batch_sum:
            return torch.sum(output, dim=1)
        else:
            return output

