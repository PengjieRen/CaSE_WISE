import torch.nn as nn
import torch.nn.functional as F
import torch

class T1(nn.Module):
    def __init__(self, embedding, t1_encoder, hidden_size, id2intent):
        super(T1, self).__init__()

        self.hidden_size=hidden_size
        self.id2intent=id2intent

        self.embedding=embedding
        self.t1_encoder=t1_encoder

        self.linear = nn.Linear(hidden_size, len(id2intent), bias=False)

    def label(self, t1_output):
        # t1_output [batch_size], translate intent_id to the corresponding intent
        # format of returned data is ["intent1", "intent2", ...]
        return [self.id2intent[t1_output[i].item()] for i in range(t1_output.size(0))]

    def forward(self, context, common_output):
        # context_emb [batch_size, context_len, hidden_size]
        context_emb = self.embedding(context)

        # context_states 4 * [batch_size, context_len, hidden_size]
        # context_weights 4 * [batch_size, context_len, context_len]
        context_states, context_weights = self.t1_encoder(context_emb, src_key_padding_mask=context.eq(0))
        # -1 means the last layer of transformers
        # parameters [batch_size, hidden_size], 0 means I-CLS of context
        # t1_output [batch_size, num_intent]
        t1_output = self.linear(context_states[-1][:, 0])

        common_output['context_states'] = context_states
        common_output['context_weights'] = context_weights
        common_output['t1_output']=t1_output

        return common_output

    def loss(self, t1_output, intent):
        t1_loss = F.cross_entropy(t1_output, intent.reshape(-1))
        return t1_loss


