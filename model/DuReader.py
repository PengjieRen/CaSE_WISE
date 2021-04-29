import torch.nn as nn
import torch
import copy

class DuReaderModel(nn.Module):
    def __init__(self, t1, t2, t3, t4, t5, t6, response_len=50):
        super(DuReaderModel, self).__init__()

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

        if 't5_output' not in common_output:
            self.t5(context, passage, common_output)

        if response is not None:
            self.t6(context, query, passage, response, common_output)

        return common_output

    def forward(self, data, method):
        common_output = {'selected_passage': data['selected_passage'], 'method': 'train'}
        output = self.do_forward(data['context'], None, data['passage_candidate'], data['response'][:, :-1], common_output)
        t1_loss = self.t1.loss(output['t1_output'], data['intent'])
        t2_loss = self.t2.loss(output['t2_output'], data['state'], data['state_loss_mask'])
        t3_loss = self.t3.loss(output['t3_output'], data['action'])
        t5_loss = self.t5.loss(output['t5_output'], data['selected_passage'], data['passage_loss_mask'])
        t6_loss = self.t6.loss(output['t6_output'], data['response'])
        return {'t1_loss': t1_loss, 't2_loss': t2_loss, 't3_loss': t3_loss, 't5_loss': t5_loss, 't6_loss': t6_loss}
        # return {'t5_loss': t5_loss}





