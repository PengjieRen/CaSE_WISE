import codecs
import json
import copy
import argparse
import os
import re
from common.Utils import *
from torch.utils.data import Dataset
from data.Utils import *
import torch
dir_path = os.path.dirname(os.path.realpath(__file__))

class KdConvDataset(Dataset):
    def __init__(self, conv_files, doc_files, vocab2id, tokenizer, context_len=100, query_len=50, num_queries=10, response_len=50):
        super(KdConvDataset, self).__init__()
        self.vocab2id=vocab2id
        self.tokenizer=tokenizer
        self.context_len=context_len
        self.query_len=query_len
        self.num_queries=num_queries
        self.response_len=response_len

        self.passages = {}
        self.samples=[]
        if os.path.exists(os.path.join(dir_path, 'KdConv.conversation.pkl')):
            self.samples=torch.load(os.path.join(dir_path, 'KdConv.conversation.pkl'))
        else:
            if os.path.exists(os.path.join(dir_path, 'KdConv.document.pkl')):
                self.passages = torch.load(os.path.join(dir_path, 'KdConv.document.pkl'))
            else:
                for file in doc_files:
                    self.passages = load_passage(file, tokenizer, self.passages)
                if not os.path.exists(os.path.join(dir_path, 'KdConv.document.pkl')):
                    torch.save(self.passages, os.path.join(dir_path, 'KdConv.document.pkl'))

            for file in conv_files:
                self.load(file)
            if not os.path.exists(os.path.join(dir_path, 'KdConv.conversation.pkl')):
                torch.save(self.samples, os.path.join(dir_path, 'KdConv.conversation.pkl'))

        self.len=len(self.samples)

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

                state = conv[-1]['state']
                state_tensor = torch.zeros((len(context),)).long()
                build_state_tensor(context, state, state_tensor, self.tokenizer)

                query_candidate = copy.deepcopy(conv[-1]['selected_query'])[:self.num_queries]
                random.shuffle(conv[-1]['query_candidate'])
                if len(query_candidate) < self.num_queries:
                    for query in conv[-1]['query_candidate']:
                        if query not in query_candidate:
                            query_candidate.append(query)
                            if len(query_candidate) == self.num_queries:
                                break
                random.shuffle(query_candidate)
                query_candidate_tensor = []
                selected_query = conv[-1]['selected_query']
                selected_query_tensor = []
                for i in range(len(query_candidate)):
                    query = ['[Q-CLS]'] + self.tokenizer(query_candidate[i])
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

                # fill query_candidate
                while len(query_candidate_tensor) < self.num_queries:
                    query = ['[Q-CLS]'] + ['[PAD]'] * (self.query_len - 1)
                    query_tensor = torch.tensor([self.vocab2id.get(w, self.vocab2id['[UNK]']) for w in query]).long()
                    query_candidate_tensor.append(query_tensor)
                    selected_query_tensor.append(0)
                query_candidate_tensor = torch.stack(query_candidate_tensor)
                selected_query_tensor = torch.tensor(selected_query_tensor).long()

                # add action label in the start of response, and EOS in the end of response
                response = ['[Conv]']+self.tokenizer(conv[-1]['response'])[:self.response_len]+['[EOS]']
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

                self.samples.append(((id, context, state, query_candidate, selected_query, response), (id_tensor, context_tensor, state_tensor, query_candidate_tensor, selected_query_tensor, response_tensor, state_loss_mask, query_loss_mask)))

    def __getitem__(self, index):
        return self.samples[index][1]

    def __len__(self):
        return self.len

def kdconv_collate_fn(data):
    id_tensor, context_tensor, state_tensor, query_candidate_tensor, selected_query_tensor, response_tensor, state_loss_mask, query_loss_mask = zip(*data)
    # id [batch_size]
    # context [batch_size, context_len]
    # state [batch_size, context_len]
    # query_candidate [batch_size, num_queries, query_len]
    # selected_query [batch_size, num_queries]  0/1 sequence
    # response [batch_size, response_len]
    # state_loss_mask/query_loss_mask [batch_size, 1]  0/1 sequence
    return {'id': torch.cat(id_tensor),
            'context': torch.stack(context_tensor),
            'state': torch.stack(state_tensor),
            'query_candidate': torch.stack(query_candidate_tensor),
            'selected_query': torch.stack(selected_query_tensor),
            'response': torch.stack(response_tensor),
            'state_loss_mask': torch.stack(state_loss_mask),
            'query_loss_mask': torch.stack(query_loss_mask)
            }


def prepare_kdconv_dataset(args):
    if os.path.exists(args.kdconv_conversation_file):
        return
    msg_id=1
    document_id=1

    cfile = codecs.open(args.kdconv_conversation_file, "w", "utf-8")
    dfile = codecs.open(args.kdconv_document_file, "w", "utf-8")
    for file in args.kdconv_files:
        print(file)
        with open(file, encoding='utf-8') as f:
            convs=json.loads(f.read())

            for conv in convs:
                conversations = []
                passage_id = 1
                passages = []
                candidates = []
                for item in conv['messages']:
                    if 'attrs' in item:
                        # 知识作为passage
                        sentences = [{'id': i+1, 'text':item['attrs'][i]['name']+'-'+item['attrs'][i]['attrname']+'-'+item['attrs'][i]['attrvalue']} for i in range(len(item['attrs']))]
                        candidates.append(str(document_id)+'-'+str(passage_id))
                        selected=[str(document_id)+'-'+str(passage_id)]
                        passages.append({'id': passage_id, 'sentences': sentences})
                        passage_id+=1
                        states=set()
                        context=''.join([c['response'] for c in conversations])
                        for sent in sentences:
                            states = states.union(extract_state(context, sent['text']))
                        states = states.union(extract_state(context, item['message']))
                        conversations.append({'msg_id':msg_id, 'turn_id': 1, 'role':'system', 'state':list(states), 'query_candidate':candidates, 'selected_query':selected, 'response': item['message']})
                    else:
                        conversations.append({'msg_id':msg_id, 'turn_id': 1, 'role':'user', 'response':item['message']})

                dfile.write(json.dumps({'id': document_id, 'passages': passages}, ensure_ascii=False) + os.linesep)
                document_id += 1

                for i in range(len(conversations)):
                    if conversations[i]['role'] == 'system':
                        sample = copy.deepcopy(conversations)[:i + 1]
                        cfile.write(json.dumps(sample, ensure_ascii=False) + os.linesep)
    cfile.close()
    dfile.close()

