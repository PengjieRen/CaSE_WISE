import torch
import torch.nn.functional as F
from torch.nn.modules.module import Module
import torch.nn as nn
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm
import copy
from torch.nn.modules.container import ModuleList

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    else:
        raise RuntimeError("activation should be relu/gelu, not %s." % activation)

class TransformerEncoderLayer(Module):
    def __init__(self, hidden_size, num_heads, dim_feedforward=512, dropout=0.1, activation="relu"):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout)
        self.linear1 = Linear(hidden_size, dim_feedforward)
        self.linear2 = Linear(dim_feedforward, hidden_size, bias=False)

        self.norm1 = LayerNorm(hidden_size)
        self.norm2 = LayerNorm(hidden_size)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src = src.transpose(0, 1)

        src = self.norm1(src)
        src2, weight = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)

        src = self.norm2(src)
        if hasattr(self, "activation"):
            src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        else:  # for backward compatibility
            src2 = self.linear2(self.dropout2(F.relu(self.linear1(src))))
        src = src + self.dropout3(src2)

        # src.transpose(0,1) [batch_size*num_seq, seq_len, hidden_size]
        return src.transpose(0, 1), weight

def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])

class TransformerEncoder(Module):
    def __init__(self, encoder_layer, num_layers):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        output = src

        outputs=[]
        weights=[]
        for i in range(self.num_layers):
            output, weight = self.layers[i](output, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)
            outputs.append(output)
            weights.append(weight)

        return outputs, weights