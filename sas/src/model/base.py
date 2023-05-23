import sys
import torch
import torch.nn.functional as F

sys.path.append("..")


class Attention(torch.nn.Module):
    def __init__(self, config, input_dim):
        super(Attention, self).__init__()
        self.config = config

        self.q_w = torch.nn.Linear(input_dim, self.config.attention_hidden_size, bias=False)
        self.k_w = torch.nn.Linear(input_dim, self.config.attention_hidden_size, bias=False)
        self.v_w = torch.nn.Linear(input_dim, self.config.attention_hidden_size, bias=False)
        self.shrink = torch.nn.Linear(self.config.attention_hidden_size, 1)

        self.tanh = torch.nn.Tanh()

    def forward(self, query_tensor, input_tensor, shrink=True):
        # Q, K, V
        query_tensor = self.q_w(query_tensor)
        key_tensor = self.k_w(input_tensor)
        value_tensor = self.v_w(input_tensor)
        # ATTN = softmax(QK^T)
        attn_weight = torch.bmm(query_tensor, torch.transpose(key_tensor, 1, 2))
        attn_weight = F.softmax(attn_weight, dim=2)
        # ATTN x V
        output = torch.bmm(attn_weight, value_tensor)
        output = self.tanh(output)
        # squeeze
        output, attn_weight = output.squeeze(1), attn_weight.squeeze(1)
        if shrink:
            output = self.shrink(output)
        return output, attn_weight


class ModelBase(torch.nn.Module):
    def __init__(self, config):
        super(ModelBase, self).__init__()
        self.config = config

        # library layer
        self.dropout = torch.nn.Dropout(self.config.dropout_rate)
        self.relu = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()
