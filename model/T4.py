import torch.nn as nn
import torch.nn.functional as F
import torch
from model.Utils import *

class T4(nn.Module):
    def __init__(self, embedding, t4_encoder, hidden_size):
        super(T4, self).__init__()

        self.hidden_size=hidden_size

        self.embedding=embedding
        self.t4_encoder=t4_encoder

        self.linear = nn.Linear(hidden_size, 1, bias=False)

    def label(self, t4_output):
        # t4_output [batch_size, num_queries], 0/1 sequence. A value of 1 indicates that the corresponding query is selected by system
        # format of return [[q11, q12, ...],
        #                   [q21, q22, ...],
        #                   ...]
        selected=[]
        for i in range(t4_output.size(0)):
            rs=[]
            for j in range(t4_output.size(1)):
                if t4_output[i,j]:
                    rs.append(j)
            selected.append(rs)
        return selected

    def forward(self, context, query, common_output):
        # query_context_states 4 * [batch_size, num_queries, query_len + context_len, hidden_size]
        # paramaters [batch_size, num_queries, hidden_size], 0 means Q-CLS of query
        # t4_output [batch_size, num_queries]
        t4_output = self.linear(common_output['query_context_states'][-1][:, :, 0]).squeeze(-1)
        common_output['t4_output'] = t4_output

        return common_output

    def loss(self, t4_output, selected_query, query_loss_mask):
        # query_loss_mask [batch_size, 1]
        # t4_output&selected_query [batch_size, num_queries]
        t4_loss = (query_loss_mask.detach() * F.binary_cross_entropy_with_logits(t4_output, selected_query.float(), reduction='none').mean(dim=1, keepdim=True) + 1e-8).sum()/(query_loss_mask.detach().sum()+1)
        return t4_loss



