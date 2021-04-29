import torch.nn as nn
import torch
import copy

class WISE_T1Model(nn.Module):
    def __init__(self, t1, t2, t3, t4, t5, t6, response_len=50):
        super(WISE_T1Model, self).__init__()

        self.response_len=response_len

        self.t1 = t1
        self.t2 = t2
        self.t3 = t3
        self.t4 = t4
        self.t5 = t5
        self.t6 = t6

    def intent(self, t1_output):
        return self.t1.label(t1_output)

    def state(self, t2_output):
        return self.t2.label(t2_output)

    def action(self, t3_output):
        return self.t3.label(t3_output)

    def query(self, t4_output):
        return self.t4.label(t4_output)

    def passage(self, t5_output):
        return self.t5.label(t5_output)

    def response(self, t6_output):
        return self.t6.label(t6_output)

    def do_forward(self, context, query, passage, response, common_output):

        if 't1_output' not in common_output:
            self.t1(context, common_output)

        if 't2_output' not in common_output:
            self.t2(context, common_output)

        if 't3_output' not in common_output:
            self.t3(context, query, passage, common_output)

        if 't4_output' not in common_output:
            self.t4(context, query, common_output)

        if 't5_output' not in common_output:
            self.t5(context, passage, common_output)

        if response is not None:
            self.t6(context, query, passage, response, common_output)

        return common_output

    def forward(self, data, method):
        if method=='train':
            common_output = {'selected_query': data['selected_query'], 'selected_passage': data['selected_passage'], 'method': 'train'}
            output=self.do_forward(data['context'], data['query_candidate'], data['passage_candidate'], data['response'][:, :-1], common_output)
            # t2_output [batch_size, context_len]    state [batch_size, context_len]    state_loss_mask [batch_size, 1]
            t2_loss = self.t2.loss(output['t2_output'], data['state'], data['state_loss_mask'])
            # t3_output [batch_size, num_actions]    action [batch_size, 1]
            t3_loss = self.t3.loss(output['t3_output'], data['action'])
            # t4_output [batch_size, num_queries]    selected_query [batch_size, queries]    query_loss_mask [batch_size, 1]
            t4_loss = self.t4.loss(output['t4_output'], data['selected_query'], data['query_loss_mask'])
            # t5_output [batch_size, num_passages]    selected_passage [batch_size, passages]    passage_loss_mask [batch_size, 1]
            t5_loss = self.t5.loss(output['t5_output'], data['selected_passage'], data['passage_loss_mask'])
            # t6_output [batch_size, response_len - 1, vocab_size]    response [batch_size, response_len]
            t6_loss = self.t6.loss(output['t6_output'], data['response'])
            return {'t2_loss':t2_loss, 't3_loss':t3_loss, 't4_loss':t4_loss, 't5_loss':t5_loss, 't6_loss':t6_loss}
        elif method == 'test':
            common_output = {'method': 'test'}
            output = self.do_forward(data['context'], data['query_candidate'], data['passage_candidate'], None, common_output)
            # bos [batch_size, 1]
            bos=output['t3_output'].argmax(dim=-1, keepdim=True)+30
            # use ground truth of action
            # bos = data['action'] + 30
            response=bos
            for i in range(self.response_len-1):
                # response [batch_size, active_response_len]
                output = self.do_forward(data['context'], data['query_candidate'], data['passage_candidate'], response, output)
                # t6_output [batch_size, active_response_len, vocab_size] --> [batch_size, active_response_len]
                # bos cat t6_output [batch_size, response_len + 1]
                # reponse [batch_size, active_response_len + 1]
                response = torch.cat([bos, output['t6_output'].argmax(dim=-1, keepdim=False)], dim=-1)
            return_output={}
            # t1_output [batch_size, num_intents] --> [batch_size]
            return_output['t1_output'] = output['t1_output'].argmax(dim=-1, keepdim=False)
            # t2_output [batch_size, context_len]
            return_output['t2_output'] = copy.deepcopy(data['context']).masked_fill_(torch.sigmoid(output['t2_output']) < 0.5, 0)
            # t3_output [batch_size, num_actions] --> [batch_size]
            return_output['t3_output'] = output['t3_output'].argmax(dim=-1, keepdim=False)
            # t4_output [batch_size, num_queries]
            return_output['t4_output'] = torch.sigmoid(output['t4_output']) > 0.5
            # t5_output [batch_size, num_passages]
            return_output['t5_output'] = torch.sigmoid(output['t5_output']) > 0.5
            # t6_output [batch_size, response_len - 1, vocab_size] --> [batch_size, response_len - 1]
            return_output['t6_output'] = output['t6_output'].argmax(dim=-1, keepdim=False)

            return return_output





