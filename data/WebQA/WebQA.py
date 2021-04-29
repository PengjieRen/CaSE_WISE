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

class WebQADataset(Dataset):
    def __init__(self, conv_files, doc_files, vocab2id, tokenizer, context_len=50, passage_len=100, num_passages=10, response_len=10):
        super(WebQADataset, self).__init__()
        self.vocab2id=vocab2id
        self.tokenizer=tokenizer
        self.context_len=context_len
        self.passage_len=passage_len
        self.num_passages=num_passages
        self.response_len=response_len

        self.passages = {}
        self.samples=[]
        if os.path.exists(os.path.join(dir_path, 'WebQA.conversation.pkl')):
            self.samples=torch.load(os.path.join(dir_path, 'WebQA.conversation.pkl'))
        else:
            if os.path.exists(os.path.join(dir_path, 'WebQA.document.pkl')):
                self.passages = torch.load(os.path.join(dir_path, 'WebQA.document.pkl'))
            else:
                for file in doc_files:
                    self.passages = load_passage(file, tokenizer, self.passages)
                if not os.path.exists(os.path.join(dir_path, 'WebQA.document.pkl')):
                    torch.save(self.passages, os.path.join(dir_path, 'WebQA.document.pkl'))

            for file in conv_files:
                self.load(file)
            if not os.path.exists(os.path.join(dir_path, 'WebQA.conversation.pkl')):
                torch.save(self.samples, os.path.join(dir_path, 'WebQA.conversation.pkl'))

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

                passage_candidate = copy.deepcopy(conv[-1]['selected_passage'])[:self.num_passages]
                random.shuffle(conv[-1]['passage_candidate'])
                if len(passage_candidate) < self.num_passages:
                    for passage in conv[-1]['passage_candidate']:
                        if passage not in passage_candidate:
                            passage_candidate.append(passage)
                            if len(passage_candidate)==self.num_passages:
                                break
                random.shuffle(passage_candidate)
                passage_candidate_tensor = []
                selected_passage = conv[-1]['selected_passage']
                selected_passage_tensor = []
                # add P-CLS in the start of passage
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
                # fill passage_candidate
                while len(passage_candidate_tensor) < self.num_passages:
                    passage = ['[P-CLS]'] + ['[PAD]'] * (self.passage_len - 1)
                    passage_tensor = torch.tensor(
                        [self.vocab2id.get(w, self.vocab2id['[UNK]']) for w in passage]).long()
                    passage_candidate_tensor.append(passage_tensor)
                    selected_passage_tensor.append(0)
                passage_candidate_tensor = torch.stack(passage_candidate_tensor)
                selected_passage_tensor = torch.tensor(selected_passage_tensor).long()

                # add action label in the start of response, and EOS in the end of response
                response = ['[QA]']+self.tokenizer(conv[-1]['response'])[:self.response_len]+['[EOS]']
                if len(response) < self.response_len+2:
                    response += ['[PAD]'] * (self.response_len+2 - len(response))
                response_tensor = torch.tensor([self.vocab2id.get(w, self.vocab2id['[UNK]']) for w in response]).long()

                if len(selected_passage)>0:
                    passage_loss_mask = torch.tensor([1]).long()
                else:
                    passage_loss_mask = torch.tensor([0]).long()

                self.samples.append(((id, context, passage_candidate, selected_passage, response), (id_tensor, context_tensor, passage_candidate_tensor, selected_passage_tensor, response_tensor, passage_loss_mask)))

    def __getitem__(self, index):
        return self.samples[index][1]

    def __len__(self):
        return self.len

def webqa_collate_fn(data):
    id_tensor, context_tensor, passage_candidate_tensor, selected_passage_tensor, response_tensor, passage_loss_mask = zip(*data)
    # id [batch_size]
    # context [batch_size, context_len]
    # selected_passage [batch_size, num_passages]  0/1 sequence
    # response [batch_size, response_len]
    return {'id': torch.cat(id_tensor),
            'context': torch.stack(context_tensor),
            'passage_candidate': torch.stack(passage_candidate_tensor),
            'selected_passage': torch.stack(selected_passage_tensor),
            'response': torch.stack(response_tensor),
            'passage_loss_mask': torch.stack(passage_loss_mask)
            }


def prepare_webqa_dataset(args):
    if os.path.exists(args.webqa_conversation_file):
        return

    document_id=1
    msg_id=1

    cfile = codecs.open(args.webqa_conversation_file, "w", "utf-8")
    dfile = codecs.open(args.webqa_document_file, "w", "utf-8")
    for file in args.webqa_files:
        print(file)
        with codecs.open(file, encoding='utf-8') as f:
            convs=json.loads(f.read())
            for key in convs:
                conv=convs[key]
                user={'msg_id':msg_id, 'turn_id': 1, 'role':'user', 'response':conv['question']}
                msg_id+=1
                candidates=[]
                passage_id=1
                passages=[]
                for e in conv['evidences']:
                    evidence=conv['evidences'][e]

                    for answer in evidence['answer']:
                        if answer=='no_answer':
                            candidates.append(str(document_id) + '-' + str(passage_id))

                            sents = re.split('(。|！|\!|？|\?)', evidence['evidence'])
                            sentences = []
                            for i in range(len(sents)):
                                if len(sents[i]) > 1:
                                    sentences.append({'id': i + 1, 'text': sents[i]})
                            passages.append({'id': passage_id, 'sentences': sentences})
                            passage_id += 1

                for e in conv['evidences']:
                    evidence=conv['evidences'][e]

                    for answer in evidence['answer']:
                        if answer!='no_answer':
                            system = {'msg_id'
                                      : msg_id, 'turn_id': 1, 'role': 'system', 'passage_candidate': candidates+[str(document_id)+'-'+str(passage_id)], 'selected_passage': [str(document_id)+'-'+str(passage_id)], 'response': answer}
                            msg_id += 1
                            cfile.write(json.dumps([user, system], ensure_ascii=False) + os.linesep)

                            sents = re.split('(。|！|\!|？|\?)', evidence['evidence'])
                            sentences = []
                            for i in range(len(sents)):
                                if len(sents[i]) > 1:
                                    sentences.append({'id': i + 1, 'text': sents[i]})
                            passages.append({'id': passage_id, 'sentences': sentences})
                            passage_id += 1

                dfile.write(json.dumps({'id':document_id, 'passages':passages}, ensure_ascii=False)+os.linesep)
                document_id+=1

    cfile.close()
    dfile.close()


