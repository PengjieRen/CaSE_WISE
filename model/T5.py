import torch.nn as nn
import torch.nn.functional as F
import torch
from model.Utils import *

class T5(nn.Module):
    def __init__(self, embedding, t5_encoder, hidden_size):
        super(T5, self).__init__()

        self.hidden_size=hidden_size

        self.embedding=embedding
        self.t5_encoder=t5_encoder

        self.linear = nn.Linear(hidden_size, 1, bias=False)

    def label(self, t5_output):
        # t5_output [batch_size, num_passages], 0/1 sequence, a value of 1 indicates that the corresponding passage is selected by system
        # format of return [[p11, p12, ...],
        #                   [p21, p22, ...],
        #                   ...]
        selected=[]
        for i in range(t5_output.size(0)):
            rs=[]
            for j in range(t5_output.size(1)):
                if t5_output[i,j]:
                    rs.append(j)
            selected.append(rs)
        return selected

    def forward(self, context, passage, common_output):
        # passage_context_states 4 * [batch_size, num_queries, passage_len + context_len, hidden_size]
        # paramater [batch_size, num_passages, hidden_size], 0 means P-CLS of passage
        # t5_output [batch_size, num_passages]
        t5_output = self.linear(common_output['passage_context_states'][-1][:, :, 0]).squeeze(-1)

        common_output['t5_output'] = t5_output

        return common_output

    def loss(self, t5_output, selected_passage, passage_loss_mask):
        # passage_loss_mask [batch_size, 1]
        # t5_output&selected_passage [batch_size, num_passages]
        t5_loss = (passage_loss_mask.detach() * F.binary_cross_entropy_with_logits(t5_output, selected_passage.float(), reduction='none').mean(dim=1, keepdim=True) + 1e-8).sum()/(passage_loss_mask.detach().sum()+1)
        return t5_loss



