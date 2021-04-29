import os
import codecs
import json
import argparse
import copy
from common.Utils import *
from torch.utils.data import Dataset
from data.Utils import *
import torch
dir_path = os.path.dirname(os.path.realpath(__file__))

class WISEDataset(Dataset):
    def __init__(self, conv_files, doc_files, vocab2id, intent2id, action2id, tokenizer, context_len=200, query_len=10, num_queries=10, passage_len=100, num_passages=10, response_len=50):
        super(WISEDataset, self).__init__()
        self.vocab2id=vocab2id
        self.intent2id=intent2id
        self.action2id=action2id
        self.tokenizer=tokenizer
        self.context_len=context_len
        self.query_len=query_len
        self.num_queries=num_queries
        self.passage_len=passage_len
        self.num_passages=num_passages
        self.response_len=response_len

        self.passages = {}
        self.samples=[]

        flag_name = ['train', 'valid', 'testunseen', 'testseen', 'test']
        file_name = 'WISE.conversation.pkl'
        for fn in flag_name:
            if fn in conv_files[0]:
                file_name = ''.join(['WISE.conversation.', fn, '.pkl'])
                break

        if os.path.exists(os.path.join(dir_path, file_name)):
            print("load... ", file_name)
            self.samples=torch.load(os.path.join(dir_path, file_name))
        else:
            if os.path.exists(os.path.join(dir_path, 'WISE.document.pkl')):
                self.passages = torch.load(os.path.join(dir_path, 'WISE.document.pkl'))
            else:
                for file in doc_files:
                    self.passages = load_passage(file, tokenizer, self.passages)
                if not os.path.exists(os.path.join(dir_path, 'WISE.document.pkl')):
                    torch.save(self.passages, os.path.join(dir_path, 'WISE.document.pkl'))

            for file in conv_files:
                self.load(file)
            if not os.path.exists(os.path.join(dir_path, file_name)):
                print("save... ", file_name)
                torch.save(self.samples, os.path.join(dir_path, file_name))

        self.len=len(self.samples)

        self.id_samples={}
        for sample in self.samples:
            # {id: text_sample}
            self.id_samples[sample[0][0]]=sample[0]

    def load(self, file):
        with codecs.open(file, encoding='utf-8') as f:
            for line in f:
                conv = json.loads(line)
                id = conv[-1]['msg_id']
                id_tensor = torch.tensor([id]).long()
                # add I-CLS and A-CLS to context
                context = prepare_context(conv[:-1], self.tokenizer)
                if len(context)<self.context_len:
                    context+=['[PAD]']*(self.context_len-len(context))
                else:
                    context = context[:self.context_len]
                context_tensor = torch.tensor([self.vocab2id.get(w, self.vocab2id['[UNK]']) for w in context]).long()

                # 获取最后一轮（当前轮）用户话语的intent
                for i in range(len(conv)):
                    if conv[-i-1]['role']=='user':
                        intent = '[' + '-'.join(conv[-i-1]['intent']) + ']'
                        break
                intent_tensor = torch.tensor([self.intent2id[intent]]).long()

                # state of system response
                state = conv[-1]['state']
                state_tensor = torch.zeros((len(context),)).long()
                build_state_tensor(context, state, state_tensor, self.tokenizer)

                query_candidate=copy.deepcopy(conv[-1]['selected_query'])[:self.num_queries]
                if len(query_candidate)<self.num_queries:
                    for query in conv[-1]['query_candidate']:
                        if query not in query_candidate:
                            query_candidate.append(query)
                            if len(query_candidate) == self.num_queries:
                                break
                random.shuffle(query_candidate)
                query_candidate_tensor = []
                selected_query = conv[-1]['selected_query']
                selected_query_tensor = []
                # add Q-CLS to each query
                for i in range(len(query_candidate)):
                    query = [ '[Q-CLS]'] + self.tokenizer(query_candidate[i])
                    if len(query) < self.query_len:
                        query += ['[PAD]'] * (self.query_len - len(query))
                    else:
                        query = query[:self.query_len]
                    query_tensor = torch.tensor([self.vocab2id.get(w, self.vocab2id['[UNK]']) for w in query]).long()
                    query_candidate_tensor.append(query_tensor)
                    if query_candidate[i] in selected_query:
                        selected_query_tensor.append(1)
                    else:
                        selected_query_tensor.append(0)
                while len(query_candidate_tensor) < self.num_queries:
                    query = [ '[Q-CLS]'] + ['[PAD]'] * (self.query_len - 1)
                    query_tensor = torch.tensor([self.vocab2id.get(w, self.vocab2id['[UNK]']) for w in query]).long()
                    query_candidate_tensor.append(query_tensor)
                    selected_query_tensor.append(0)
                query_candidate_tensor = torch.stack(query_candidate_tensor)
                selected_query_tensor = torch.tensor(selected_query_tensor).long()

                if len(conv[-1]['selected_passage']) > 0 and not conv[-1]['selected_passage'][0].split('-')[0]:
                    passage_candidate = []
                else:
                    passage_candidate = copy.deepcopy(conv[-1]['selected_passage'])[:self.num_passages]
                if len(passage_candidate) < self.num_passages:
                    for passage in conv[-1]['passage_candidate']:
                        if passage not in passage_candidate:
                            passage_candidate.append(passage)
                            if len(passage_candidate) == self.num_passages:
                                break
                random.shuffle(passage_candidate)
                passage_candidate_tensor = []
                selected_passage = conv[-1]['selected_passage']
                selected_passage_tensor = []
                # add P-CLS to each passage
                for i in range(len(passage_candidate)):
                    passage = ['[P-CLS]'] + self.passages[passage_candidate[i]]
                    if len(passage) < self.passage_len:
                        passage += ['[PAD]'] * (self.passage_len - len(passage))
                    else:
                        passage = passage[:self.passage_len]
                    passage_tensor = torch.tensor([self.vocab2id.get(w, self.vocab2id['[UNK]']) for w in passage]).long()
                    passage_candidate_tensor.append(passage_tensor)
                    if passage_candidate[i] in selected_passage:
                        selected_passage_tensor.append(1)
                    else:
                        selected_passage_tensor.append(0)
                while len(passage_candidate_tensor) < self.num_passages:
                    passage = ['[P-CLS]'] + ['[PAD]'] * (self.passage_len - 1)
                    passage_tensor = torch.tensor([self.vocab2id.get(w, self.vocab2id['[UNK]']) for w in passage]).long()
                    passage_candidate_tensor.append(passage_tensor)
                    selected_passage_tensor.append(0)
                passage_candidate_tensor = torch.stack(passage_candidate_tensor)
                selected_passage_tensor = torch.tensor(selected_passage_tensor).long()

                # action of system response
                action = '[' + '-'.join(conv[-1]['action']) + ']'
                action_tensor = torch.tensor([self.action2id[action]]).long()

                # add action label in the start of response, and EOS in the end of response
                response = ['['+'-'.join(conv[-1]['action'])+']']+self.tokenizer(conv[-1]['response'])[:self.response_len]+['[EOS]']
                if len(response) < self.response_len+2:
                    response += ['[PAD]'] * (self.response_len+2 - len(response))
                response_tensor = torch.tensor([self.vocab2id.get(w, self.vocab2id['[UNK]']) for w in response]).long()

                if len(state)>0:
                    state_loss_mask = torch.tensor([1]).long()
                else:
                    state_loss_mask = torch.tensor([0]).long()
                if len(selected_query)>0:
                    query_loss_mask=torch.tensor([1]).long()
                else:
                    query_loss_mask = torch.tensor([0]).long()

                if len(selected_passage)>0:
                    passage_loss_mask=torch.tensor([1]).long()
                else:
                    passage_loss_mask = torch.tensor([0]).long()

                self.samples.append(((id, context, intent, state, query_candidate, passage_candidate, action, selected_query, selected_passage, response), (id_tensor, context_tensor, intent_tensor, state_tensor, query_candidate_tensor, passage_candidate_tensor, action_tensor, selected_query_tensor, selected_passage_tensor, response_tensor, state_loss_mask, query_loss_mask, passage_loss_mask)))

    def __getitem__(self, index):
        return self.samples[index][1]

    def __len__(self):
        return self.len

    def query(self, id, index):
        if index<len(self.id_samples[id][4]):
            return self.id_samples[id][4][index]
        else:
            return 'None'

    def passage(self, id, index):
        if index<len(self.id_samples[id][5]):
            return self.id_samples[id][5][index]
        else:
            return 'None'


def wise_collate_fn(data):
    id_tensor, context_tensor, intent_tensor, state_tensor, query_candidate_tensor, passage_candidate_tensor, action_tensor, selected_query_tensor, selected_passage_tensor, response_tensor, state_loss_mask, query_loss_mask, passage_loss_mask = zip(*data)
    # id [batch_size]
    # context [batch_size, context_len]
    # intent [batch_size, 1]
    # state [batch_size, context_len]
    # query_candidate [batch_size, num_queries, query_len]
    # passage_candidate [batch_size, num_passages, passage_len]
    # action [batch_size, num_actions]
    # selected_query [batch_size, num_queries]  0/1 sequence
    # selected_passage [batch_size, num_passages]  0/1 sequence
    # response [batch_size, response_len]
    # state_loss_mask/query_loss_mask/passage_loss_mask [batch_size, 1]  0/1 sequence
    return {'id': torch.cat(id_tensor),
            'context': torch.stack(context_tensor),
            'intent': torch.stack(intent_tensor),
            'state': torch.stack(state_tensor),
            'query_candidate': torch.stack(query_candidate_tensor),
            'passage_candidate': torch.stack(passage_candidate_tensor),
            'action': torch.stack(action_tensor),
            'selected_query': torch.stack(selected_query_tensor),
            'selected_passage': torch.stack(selected_passage_tensor),
            'response': torch.stack(response_tensor),
            'state_loss_mask':torch.stack(state_loss_mask),
            'query_loss_mask': torch.stack(query_loss_mask),
            'passage_loss_mask': torch.stack(passage_loss_mask)
            }

def prepare_wise_dataset(args):
    # 处理原始对话数据，将一段多轮对话分割成多段“单轮”对话
    if not os.path.exists(args.wise_train_conversation_file):
        train_file = codecs.open(args.wise_train_conversation_file, 'w', 'utf-8')
        with codecs.open(args.wise_train_file) as f:
            for line in f:
                convs = json.loads(line)
                for i in range(len(convs['conversations'])):
                    if convs['conversations'][i]['role'] == 'system':
                        sample = copy.deepcopy(convs['conversations'])[:i + 1]
                        train_file.write(json.dumps(sample, ensure_ascii=False) + os.linesep)
        train_file.close()

        valid_file = codecs.open(args.wise_valid_conversation_file, 'w', 'utf-8')
        with codecs.open(args.wise_valid_file) as f:
            for line in f:
                convs = json.loads(line)
                for i in range(len(convs['conversations'])):
                    if convs['conversations'][i]['role'] == 'system':
                        sample = copy.deepcopy(convs['conversations'])[:i + 1]
                        valid_file.write(json.dumps(sample, ensure_ascii=False) + os.linesep)
        valid_file.close()

        test_file = codecs.open(args.wise_test_conversation_file, 'w', 'utf-8')
        with codecs.open(args.wise_test_file) as f:
            for line in f:
                convs = json.loads(line)
                for i in range(len(convs['conversations'])):
                    if convs['conversations'][i]['role'] == 'system':
                        sample = copy.deepcopy(convs['conversations'])[:i + 1]
                        test_file.write(json.dumps(sample, ensure_ascii=False) + os.linesep)
        test_file.close()

        testunseen_file = codecs.open(args.wise_testunseen_conversation_file, 'w', 'utf-8')
        with codecs.open(args.wise_testunseen_file) as f:
            for line in f:
                convs = json.loads(line)
                for i in range(len(convs['conversations'])):
                    if convs['conversations'][i]['role'] == 'system':
                        sample = copy.deepcopy(convs['conversations'])[:i + 1]
                        testunseen_file.write(json.dumps(sample, ensure_ascii=False) + os.linesep)
        testunseen_file.close()

        testseen_file = codecs.open(args.wise_testseen_conversation_file, 'w', 'utf-8')
        with codecs.open(args.wise_testseen_file) as f:
            for line in f:
                convs = json.loads(line)
                for i in range(len(convs['conversations'])):
                    if convs['conversations'][i]['role'] == 'system':
                        sample = copy.deepcopy(convs['conversations'])[:i + 1]
                        testseen_file.write(json.dumps(sample, ensure_ascii=False) + os.linesep)
        testseen_file.close()




