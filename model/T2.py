import torch.nn as nn
import torch.nn.functional as F
import torch
import copy

class T2(nn.Module):
    def __init__(self, embedding, t2_encoder, hidden_size, id2vocab):
        super(T2, self).__init__()

        self.hidden_size=hidden_size
        self.id2vocab=id2vocab

        self.embedding=embedding
        self.t2_encoder=t2_encoder

        self.linear = nn.Linear(hidden_size, 1, bias=False)

    def label(self, t2_output):
        # -1 means the last layer of transformers
        # parameters [batch_size, hidden_size], 0 means I-CLS of context
        rs = list()
        for i in range(t2_output.size(0)):
            indexes = t2_output[i]
            keywords = []
            keyword=[]
            for index in indexes:
                index = index.item()
                w = self.id2vocab[index]
                if w.startswith('[') and w.endswith(']'):
                    if len(keyword)>0:
                        keywords.append(keyword)
                    keyword=[]
                    continue
                keyword.append(w)
            if len(keyword) > 0:
                keywords.append(keyword)
            rs.append(keywords)
        return rs

    def forward(self, context, common_output):

        # context_states 4 * [batch_size, context_len, hidden_size]
        context_states = common_output['context_states']
        # t2_output [batch_size, context_len, 1] --> [batch_size, context_len]
        t2_output = self.linear(context_states[-1]).squeeze(-1)

        common_output['t2_output'] = t2_output

        return common_output

    def loss(self, t2_output, state, state_loss_mask):
        # state_loss_mask [batch_size, 1]
        # t2_output&state [batch_size, context_len - 2], [2:] means overlooking I-CLS and A-CLS of context
        t2_loss = (state_loss_mask.detach() * F.binary_cross_entropy_with_logits(t2_output[:, 2:], state[:, 2:].float(), reduction='none').mean(dim=1, keepdim=True) + 1e-8).sum()/(state_loss_mask.detach().sum()+1)
        return t2_loss



