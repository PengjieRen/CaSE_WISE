import torch.nn as nn
import torch
import math

class PositionalEmbedding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEmbedding(d_model)
    """

    def __init__(self, embedding_size, dropout=0.1, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        self.embedding_size=embedding_size
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, embedding_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_size, 2).float() * (-math.log(10000.0) / embedding_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [batch size, *, sequence length, embed dim]
            output: [batch size, *, sequence length, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """
        p=self.pe[:x.size(-2)]
        for i in range(len(x.size())-2):
            p=p.unsqueeze(0)
        x = x*math.sqrt(self.embedding_size) + p
        return self.dropout(x)