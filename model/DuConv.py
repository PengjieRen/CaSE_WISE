import torch.nn as nn
import torch
import copy
from torch.cuda.amp import *

class DuConvModel(nn.Module):
    def __init__(self, t1, t2, t3, t4, t5, t6, response_len=50):
        super(DuConvModel, self).__init__()

        self.response_len=response_len

        self.t1 = t1
        self.t2 = t2
        self.t3 = t3
        self.t4 = t4
        self.t5 = t5
        self.t6 = t6

    def do_forward(self, context, query, passage, response, common_output):

        if 't1_output' not in common_output:
            self.t1(context, common_output)

        if 't2_output' not in common_output:
            self.t2(context, common_output)

        if 't3_output' not in common_output:
            self.t3(context, query, passage, common_output)

        if 't4_output' not in common_output:
            self.t4(context, query, common_output)

        if response is not None:
            self.t6(context, query, passage, response, common_output)

        return common_output

    def forward(self, data, method):
        common_output = {'selected_query': data['selected_query'], 'method': 'train'}
        output = self.do_forward(data['context'], data['query_candidate'], None, data['response'][:, :-1], common_output)
        t2_loss = self.t2.loss(output['t2_output'], data['state'], data['state_loss_mask'])
        t4_loss = self.t4.loss(output['t4_output'], data['selected_query'], data['query_loss_mask'])
        t6_loss = self.t6.loss(output['t6_output'], data['response'])
        return {'t2_loss': t2_loss, 't4_loss':t4_loss, 't6_loss': t6_loss}





