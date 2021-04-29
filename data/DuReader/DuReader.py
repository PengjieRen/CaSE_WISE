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


class DuReaderDataset(Dataset):
    def __init__(self, indexes, conv_files, doc_files, vocab2id, intent2id, action2id, tokenizer, context_len=50, passage_len=100, num_passages=10, response_len=50):
        super(DuReaderDataset, self).__init__()
        self.vocab2id=vocab2id
        self.intent2id=intent2id
        self.action2id=action2id
        self.tokenizer=tokenizer
        self.context_len=context_len
        self.passage_len=passage_len
        self.num_passages=num_passages
        self.response_len=response_len

        self.samples = []
        self.passages = {}
        if not os.path.exists(os.path.join(dir_path, 'DuReader.conversation.0.pkl')):
            if os.path.exists(os.path.join(dir_path, 'DuReader.document.pkl')):
                self.passages = torch.load(os.path.join(dir_path, 'DuReader.document.pkl'))
            else:
                for file in doc_files:
                    self.passages = load_passage(file, tokenizer, self.passages)
                if not os.path.exists(os.path.join(dir_path, 'DuReader.document.pkl')):
                    torch.save(self.passages, os.path.join(dir_path, 'DuReader.document.pkl'))

            for file in conv_files:
                self.load(file)

            size_each=int(len(self.samples)/2)
            for i in range(2):
                torch.save(self.samples[size_each*i:size_each*(i+1)], os.path.join(dir_path, 'DuReader.conversation.'+str(i)+'.pkl'))

        for i in indexes:
            self.samples  += torch.load(os.path.join(dir_path, 'DuReader.conversation.' + str(i) + '.pkl'))
        self.len=len(self.samples)

    def load(self, file):
        keys=list(self.passages.keys())
        with open(file, encoding='utf-8') as f:
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

                for i in range(len(conv)):
                    if conv[-i-1]['role']=='user':
                        intent = '[' + '-'.join(conv[-i-1]['intent']) + ']'
                        break
                intent_tensor = torch.tensor([self.intent2id[intent]]).long()

                state = conv[-1]['state']
                state_tensor = torch.zeros((len(context),)).long()
                build_state_tensor(context, state, state_tensor, self.tokenizer)

                passage_candidate = copy.deepcopy(conv[-1]['selected_passage'])[:self.num_passages]
                random.shuffle(conv[-1]['passage_candidate'])
                while len(passage_candidate) < self.num_passages:
                    passage = keys[random.randint(0, len(self.passages)-1)]
                    if passage not in passage_candidate:
                        passage_candidate.append(passage)

                random.shuffle(passage_candidate)
                passage_candidate_tensor = []
                selected_passage = conv[-1]['selected_passage']
                selected_passage_tensor = []
                for i in range(len(passage_candidate)):
                    passage = ['[P-CLS]'] + self.passages[passage_candidate[i]]
                    if len(passage) < self.passage_len:
                        passage += ['[PAD]'] * (self.passage_len - len(passage))
                    else:
                        passage = passage[:self.passage_len]
                    passage_tensor = torch.tensor(
                        [self.vocab2id.get(w, self.vocab2id['[UNK]']) for w in passage]).long()
                    passage_candidate_tensor.append(passage_tensor)
                    if passage_candidate[i] in selected_passage:
                        selected_passage_tensor.append(1)
                    else:
                        selected_passage_tensor.append(0)
                while len(passage_candidate_tensor) < self.num_passages:
                    passage = ['[P-CLS]'] + ['[PAD]'] * (self.passage_len - 1)
                    passage_tensor = torch.tensor(
                        [self.vocab2id.get(w, self.vocab2id['[UNK]']) for w in passage]).long()
                    passage_candidate_tensor.append(passage_tensor)
                    selected_passage_tensor.append(0)
                passage_candidate_tensor = torch.stack(passage_candidate_tensor)
                selected_passage_tensor = torch.tensor(selected_passage_tensor).long()

                action = '[' + '-'.join(conv[-1]['action']) + ']'
                action_tensor = torch.tensor([self.action2id[action]]).long()

                # add action label in the start of response, and EOS in the end of response
                response = ['[MRC]']+self.tokenizer(conv[-1]['response'])[:self.response_len]+['[EOS]']
                if len(response) < self.response_len+2:
                    response += ['[PAD]'] * (self.response_len+2 - len(response))
                response_tensor = torch.tensor([self.vocab2id.get(w, self.vocab2id['[UNK]']) for w in response]).long()

                if len(state)>0:
                    state_loss_mask = torch.tensor([1]).long()
                else:
                    state_loss_mask = torch.tensor([0]).long()

                if len(selected_passage)>0:
                    passage_loss_mask=torch.tensor([1]).long()
                else:
                    passage_loss_mask = torch.tensor([0]).long()

                self.samples.append(((id, context, intent, state, passage_candidate, action, selected_passage, response), (id_tensor, context_tensor, intent_tensor, state_tensor, passage_candidate_tensor, action_tensor, selected_passage_tensor, response_tensor, state_loss_mask, passage_loss_mask)))

    def __getitem__(self, index):
        return self.samples[index][1]

    def __len__(self):
        return self.len

def dureader_collate_fn(data):
    id_tensor, context_tensor, intent_tensor, state_tensor, passage_candidate_tensor, action_tensor, selected_passage_tensor, response_tensor, state_loss_mask, passage_loss_mask = zip(*data)
    # id [batch_size]
    # context [batch_size, context_len]
    # intent [batch_size, 1]
    # state [batch_size, context_len]
    # passage_candidate [batch_size, num_passages, passage_len]
    # action [batch_size, num_actions]
    # selected_passage [batch_size, num_passages]  0/1 sequence
    # response [batch_size, response_len]
    # passage_loss_mask [batch_size, 1]  0/1 sequence
    return {'id': torch.cat(id_tensor),
            'context': torch.stack(context_tensor),
            'intent': torch.stack(intent_tensor),
            'state': torch.stack(state_tensor),
            'passage_candidate': torch.stack(passage_candidate_tensor),
            'action': torch.stack(action_tensor),
            'selected_passage': torch.stack(selected_passage_tensor),
            'response': torch.stack(response_tensor),
            'state_loss_mask': torch.stack(state_loss_mask),
            'passage_loss_mask': torch.stack(passage_loss_mask)
            }


def prepare_dureader_dataset(args):
    if os.path.exists(args.dureader_conversation_file):
        return

    intents=set()
    actions=set()
    document_id=1
    msg_id=1

    cfile = codecs.open(args.dureader_conversation_file, "w", "utf-8")
    dfile = codecs.open(args.dureader_document_file, "w", "utf-8")
    for file in args.dureader_files:
        print(file)
        with open(file, encoding='utf-8') as f:
            for line in f:
                conv = json.loads(line)

                # the type of the question is intent and the type of the answer is action
                user = {'msg_id': msg_id, 'turn_id': 1, 'role': 'user', 'intent':[conv['question_type']], 'response': conv['question']}
                intents.add(conv['question_type'])
                actions.add(conv['fact_or_opinion'])

                selected_passage={}
                passage_candidates=[]
                states=set()
                # the common terms for questions, document titles, and answers are state
                for doc in conv['documents']:
                    if doc['is_selected']:
                        states=states.union(extract_state(conv['question'], doc['title']))

                    passages=[]
                    passage_id=1
                    for passage in doc['paragraphs']:
                        sents=re.split('(。|！|\!|？|\?)', passage)
                        sentences=[]
                        for i in range(len(sents)):
                            if len(sents[i]) > 1:
                                sentences.append({'id': i + 1, 'text': sents[i]})
                        if not doc['is_selected']:
                            passage_candidates.append(str(document_id)+'-'+str(passage_id))
                        else:
                            selected_passage[str(document_id)+'-'+str(passage_id)]=''.join([s['text'] for s in sentences])
                        passages.append({'id': passage_id, 'sentences':sentences})
                        passage_id+=1
                    dfile.write(json.dumps({'id': document_id, 'title':doc['title'], 'passages':passages}, ensure_ascii=False) + os.linesep)
                    document_id+=1

                for answer in conv['answers']:
                    if len(answer)<1:
                        continue
                    selected_p=[]
                    max_p=''
                    max=0
                    for p in selected_passage:
                        r=recall(selected_passage[p], answer)
                        if r>max:
                            max=r
                            max_p=p
                        if r>0.3:
                            selected_p.append(p)
                    if len(selected_p)==0 and max_p!='':
                        selected_p.append(max_p)

                    f_states = states.union(extract_state(conv['question'], answer))
                    system={'msg_id':msg_id, 'turn_id': 1, 'role':'system', 'state':list(f_states), 'action':[conv['fact_or_opinion']], 'passage_candidate':passage_candidates+selected_p, 'selected_passage':selected_p, 'response': answer}
                    msg_id+=1
                    cfile.write(json.dumps([user, system], ensure_ascii=False) + os.linesep)

    cfile.close()
    dfile.close()

    file = codecs.open(args.dureader_intent, "w", "utf-8")
    for item in list(intents):
        file.write('['+item+']'+os.linesep)
    file.close()

    file = codecs.open(args.dureader_action, "w", "utf-8")
    for item in list(actions):
        file.write('['+item+']' + os.linesep)
    file.close()

