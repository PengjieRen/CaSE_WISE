import argparse
import codecs
import json
from common.Utils import *

def load_vocab(file):
    vocab2id={}
    id2vocab={}
    with codecs.open(file, encoding='utf-8') as f:
        for w in f:
            w=w.strip('\n').strip('\r')
            vocab2id[w]=len(vocab2id)
            id2vocab[len(id2vocab)]=w
    return vocab2id, id2vocab

def prepare_context(conv_json, tokenizer):
    # add I-CLS and A-CLS
    num_conv = len(conv_json)

    responses = ['[I-CLS]', '[A-CLS]']
    for i in range(num_conv):
        conv = conv_json[num_conv-i-1]  # read in reverse
        if conv['role']=='user':
            responses+=['[User]']
        elif conv['role']=='system':
            responses += ['[System]']
        responses += tokenizer(conv['response'])

    return responses

def load_passage(document_json_file, tokenizer, passage_dict={}):
    with open(document_json_file, encoding='utf-8') as f:
        for line in f:
            doc = json.loads(line)
            doc_id=doc['id']
            passages=doc['passages']
            for passage in passages:
                passage_id=str(doc_id)+'-'+str(passage['id'])
                passage_dict[passage_id]=[]
                for sent in passage['sentences']:
                    passage_dict[passage_id]+=tokenizer(sent['text'])+['[SEP]']
                passage_dict[passage_id]=passage_dict[passage_id][:-1]
    return passage_dict

def jaccard(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union

def recall(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    return float(intersection) / len(list2)

def build_state_tensor(context, state, state_tensor, tokenizer):
    for s in state:
        s = tokenizer(s)
        if len(s) == 0 or s[0] == '':
            continue
        exact_match_index = []
        fuzzy_match_index = []
        max = -1
        index = -1
        for i in range(len(context)):
            if ''.join(context[i:i + len(s)]) == ''.join(s):
                exact_match_index.append((i, i + len(s)))
                continue
            for j in range(len(s)):
                jacc = jaccard(s, context[i:i + len(s) + j])
                if jacc >= max:
                    max = jacc
                    index = (i, i + len(s) + j)
                jacc = jaccard(s, context[i:i + len(s) - j])
                if jacc >= max:
                    max = jacc
                    index = (i, i + len(s) - j)
        if max >= 0.1:
            fuzzy_match_index.append(index)

        if len(exact_match_index) > 0:
            for st, ed in exact_match_index:
                state_tensor[st:ed] = 1
        else:
            for st, ed in fuzzy_match_index:
                if ed - st > 1:
                    state_tensor[st:ed] = 1
    return state_tensor


def default_cost(i,j,d):
    return 1

def small_cost(i,j,d):
    return 0.5

def huge_cost(i,j,d):
    return 100000000

def edit_matrix(a,
                b,
                sub_cost=default_cost,
                ins_cost=default_cost,
                del_cost=default_cost,
                trans_cost=default_cost):
    n = len(a)
    m = len(b)
    d = [[(0,'') for j in range(0,m+1)] for i in range(0,n+1)]
    for i in range(1,n+1):
        d[i][0] = (d[i-1][0][0] + del_cost(i,0,d), d[i-1][0][1] + 'd')
    for j in range(1,m+1):
        d[0][j] = (d[0][j-1][0] + ins_cost(0,j,d), d[0][j-1][1] + 'i')
    for i in range(1,n+1):
        for j in range(1,m+1):
            if a[i-1] == b[j-1]:
                d[i][j] = (d[i-1][j-1][0], d[i-1][j-1][1] + 'c')
            else:
                d[i][j] = min(
                    (d[i-1][j][0] + del_cost(i,j,d), d[i-1][j][1] + 'd'),
                    (d[i][j-1][0] + ins_cost(i,j,d), d[i][j-1][1] + 'i'),
                    (d[i-1][j-1][0] + sub_cost(i,j,d), d[i-1][j-1][1] + 's')
                )
                can_transpose = (
                    i > 2 and
                    j > 2 and
                    a[i-1] == b[j-2] and
                    a[i-2] == b[j-1]
                )
                if can_transpose:
                    d[i][j] = min(
                        d[i][j],
                        (d[i-2][j-2][0] + trans_cost(i,j,d), d[i-2][j-2][1] + 't')
                    )
    return d

def edit_diff(a, b, d):
    script = d[-1][-1][1]
    diff = []
    i = j = 0
    for k in range(0, len(script)):
        if script[k] == 'c':
            diff.append( ('c', a[i], b[j]) )
            i += 1
            j += 1
        elif script[k] == 's':
            diff.append( ('s', a[i], b[j]) )
            i += 1
            j += 1
        elif script[k] == 'd':
            diff.append( ('d', a[i]) )
            i += 1
        elif script[k] == 'i':
            diff.append( ('i', b[j]) )
            j += 1
        elif script[k] == 't':
            diff.append( ('t', a[i], b[j]) )
            i += 2
            j += 2
        else:
            raise Exception('Unsupported operation')
    return diff

def extract_state(source, target):
    #
    d = edit_matrix(source, target, sub_cost=huge_cost, ins_cost=default_cost, del_cost=default_cost,
                    trans_cost=huge_cost)
    edits = edit_diff(source, target, d)
    states = set()
    state = []
    for e in edits:
        if e[0] == 'c':
            state.append(e[1])
        else:
            if len(state) > 1:
                states.add(''.join(state))
            state = []
    if len(state)>1:
        states.add(''.join(state))
    return states



