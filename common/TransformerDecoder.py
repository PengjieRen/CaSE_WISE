import torch
import copy
from torch.nn.modules.container import ModuleList
import torch
import torch.nn.functional as F
from torch.nn.modules.module import Module
import torch.nn as nn
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    else:
        raise RuntimeError("activation should be relu/gelu, not %s." % activation)

class TransformerDecoderLayer(Module):
    def __init__(self, hidden_size, num_head, dim_feedforward=512, dropout=0.1, activation="relu"):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(hidden_size, num_head, dropout=dropout)
        self.memory_attn = nn.MultiheadAttention(hidden_size, num_head, dropout=dropout)

        self.self_dropout=Dropout(dropout)
        self.self_norm=LayerNorm(hidden_size)

        self.memory_dropout = Dropout(dropout)
        self.memory_norm = LayerNorm(hidden_size)

        self.norm = LayerNorm(hidden_size)
        self.linear1 = Linear(hidden_size, dim_feedforward)
        self.linear2 = Linear(dim_feedforward, hidden_size, bias=False)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        tgt = tgt.transpose(0, 1)
        memory = memory.transpose(0, 1)

        tgt = self.self_norm(tgt)
        tgt2, tgt_weight = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.self_dropout(tgt2)

        tgt = self.memory_norm(tgt)
        tgt2, memory_weight = self.memory_attn(tgt, memory, memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.memory_dropout(tgt2)

        tgt = self.norm(tgt)
        if hasattr(self, "activation"):
            tgt2 = self.linear2(self.dropout1(self.activation(self.linear1(tgt))))
        else:  # for backward compatibility
            tgt2 = self.linear2(self.dropout1(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        return tgt.transpose(0, 1), tgt_weight, memory_weight

def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])

class TransformerDecoder(Module):
    def __init__(self, decoder_layer, num_layer):
        super(TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layer)
        self.num_layers = num_layer

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        output = tgt

        outputs=[]
        output_weights=[]
        memory_weights=[]
        for i in range(self.num_layers):
            output, output_weight, memory_weight = self.layers[i](output, memory, tgt_mask=tgt_mask,
                                    memory_mask=memory_mask,
                                    tgt_key_padding_mask=tgt_key_padding_mask,
                                    memory_key_padding_mask=memory_key_padding_mask)
            outputs.append(output)
            output_weights.append(output_weight)
            memory_weights.append(memory_weight)


        return outputs, output_weights, memory_weights