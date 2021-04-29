import codecs
import json
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from evaluation.Rouge import *
from collections import defaultdict
import numpy as np
from scipy import stats
min_num = 1e-8

def evaluate_T1(rs_t1, gt_t1):
    # rs_t1 contains predicted results
    # gt_t1 contains ground truth results
    # macro
    count_correct = {}
    count_rs = {}
    count_gt = {}
    for id in gt_t1:
        gt='-'.join(gt_t1[id]['intent'])
        if gt not in count_gt:
            count_gt[gt]=1
        else:
            count_gt[gt] += 1
        if id in rs_t1:
            rs = '-'.join(rs_t1[id]['intent'])
            if rs not in count_rs:
                count_rs[rs]=1
            else:
                count_rs[rs] += 1
            if rs==gt:
                if rs not in count_correct:
                    count_correct[rs]=1
                else:
                    count_correct[rs] += 1
    precision_dict={}
    recall_dict={}
    f1_dict={}
    for intent in count_gt:
        correct=count_correct.get(intent, 0)
        p=0
        r=0
        if intent in count_rs:
            p=float(correct) / count_rs[intent]
            precision_dict[intent] = p
        if intent in count_gt:
            r=float(correct) / count_gt[intent]
            recall_dict[intent] = r
        f1_dict[intent] = 2.*(p * r / (p + r + min_num))
    precision=0
    recall=0
    f1=0
    for intent in precision_dict:
        precision+=precision_dict[intent]
    precision=precision/float(len(precision_dict))
    for intent in recall_dict:
        recall+=recall_dict[intent]
    recall=recall/float(len(recall_dict))
    for intent in f1_dict:
        f1+=f1_dict[intent]
    f1=f1/float(len(f1_dict))
    return {'t1_precision': precision_dict, 't1_recall': recall_dict, 't1_f1': f1_dict, 't1_avg_precision': precision, 't1_avg_recall':recall, 't1_avg_f1':f1}


def evaluate_T2(rs_t2, gt_t2, tokenizer):
    actions = ['Clarify', 'Answer']

    bleu_dict = {}
    rouge_dict={}
    count_dict={}
    for id in gt_t2:
        if id not in rs_t2:
            continue
        gt_action=gt_t2[id]['action']
        if gt_action[0] in actions:
            gt = [tokenizer(s) for s in gt_t2[id]['state']]
            if not gt:
                continue
            b=0
            r=0
            for s in rs_t2[id]['state']:
                b+=sentence_bleu(gt, s, weights=(1.,), smoothing_function=SmoothingFunction().method2)
                r+=sentence_rouge(gt, s)

            gt_action='-'.join(gt_action)
            if len(rs_t2[id]['state'])>0:
                b=b/len(rs_t2[id]['state'])
                r=r / len(rs_t2[id]['state'])
                if gt_action not in bleu_dict:
                    bleu_dict[gt_action] = b
                else:
                    bleu_dict[gt_action] += b
                if gt_action not in rouge_dict:
                    rouge_dict[gt_action] = r
                else:
                    rouge_dict[gt_action] += r
                if gt_action not in count_dict:
                    count_dict[gt_action] = 1
                else:
                    count_dict[gt_action] += 1

    for gt_action in bleu_dict:
        bleu_dict[gt_action] /= count_dict[gt_action]
        rouge_dict[gt_action] /= count_dict[gt_action]

    bleu=0
    rouge=0
    for action in bleu_dict:
        bleu += bleu_dict[action]
    bleu = bleu / float(len(bleu_dict) + 1)
    for action in rouge_dict:
        rouge += rouge_dict[action]
    rouge = rouge / float(len(rouge_dict) + 1)
    return {'t2_avg_bleu_1': bleu, 't2_avg_rouge_l': rouge, 't2_bleu_1': bleu_dict, 't2_rouge_l': rouge_dict}

def evaluate_T3(rs_t3, gt_t3):
    count_correct={}
    count_rs={}
    count_gt={}
    for id in gt_t3:
        gt='-'.join(gt_t3[id]['action'])
        if gt not in count_gt:
            count_gt[gt]=1
        else:
            count_gt[gt] += 1
        if id in rs_t3:
            rs = '-'.join(rs_t3[id]['action'])
            if rs not in count_rs:
                count_rs[rs] = 1
            else:
                count_rs[rs] += 1
            if rs == gt:
                if rs not in count_correct:
                    count_correct[rs] = 1
                else:
                    count_correct[rs] += 1
    precision_dict={}
    recall_dict={}
    f1_dict={}
    for action in count_gt:
        correct=count_correct.get(action, 0)
        p=0
        r=0
        if action in count_rs:
            p=float(correct) / count_rs[action]
            precision_dict[action] = p
        if action in count_gt:
            r=float(correct) / count_gt[action]
            recall_dict[action] = r
        f1_dict[action] = 2. * (p * r / (p + r + min_num))

    precision = 0
    recall = 0
    f1 = 0
    for action in precision_dict:
        precision += precision_dict[action]
    precision = precision / float(len(precision_dict) + 1)
    for action in recall_dict:
        recall += recall_dict[action]
    recall = recall / float(len(recall_dict) + 1)
    for action in f1_dict:
        f1 += f1_dict[action]
    f1 = f1 / float(len(f1_dict) + 1)
    return {'t3_precision': precision_dict, 't3_recall': recall_dict, 't3_f1': f1_dict, 't3_avg_precision': precision, 't3_avg_recall':recall, 't3_avg_f1':f1}

def evaluate_T4(rs_t4, gt_t4):
    actions=['Clarify', 'Answer']

    precision_dict = defaultdict(list)
    recall_dict = defaultdict(list)
    f1_dict = defaultdict(list)
    count_dict = {}
    for id in gt_t4:
        gt_action=gt_t4[id]['action']
        if id not in rs_t4:
            continue
        if gt_action[0] in actions and len(gt_t4[id]['selected_query'])>0:
            correct=len(set(gt_t4[id]['selected_query']).intersection(rs_t4[id]['selected_query']))
            rs=len(rs_t4[id]['selected_query'])
            gt=len(gt_t4[id]['selected_query'])
            p=0
            r=0
            gt_action='-'.join(gt_action)
            if rs > 0:
                p=float(correct)/rs
                precision_dict[gt_action].append(p)
            if gt > 0:
                r = float(correct)/gt
                recall_dict[gt_action].append(r)
            f = 2. * (p * r / (p + r + min_num))
            f1_dict[gt_action].append(f)
            # 统计每个action的个数
            if gt_action not in count_dict:
                count_dict[gt_action] = 1
            else:
                count_dict[gt_action] += 1
    precision_dict = dict(precision_dict)
    recall_dict = dict(recall_dict)
    f1_dict = dict(f1_dict)
    for gt_action in precision_dict:
        precision_dict[gt_action] = np.mean(precision_dict[gt_action])
    for gt_action in recall_dict:
        recall_dict[gt_action] = np.mean(recall_dict[gt_action])
    for gt_action in f1_dict:
        f1_dict[gt_action] = np.mean(f1_dict[gt_action])


    precision = 0
    recall = 0
    f1 = 0
    for action in precision_dict:
        precision += precision_dict[action]
    precision = precision / float(len(precision_dict) + 1)
    for action in recall_dict:
        recall += recall_dict[action]
    recall = recall / float(len(recall_dict) + 1)
    for action in f1_dict:
        f1 += f1_dict[action]
    f1 = f1 / float(len(f1_dict) + 1)

    return {'t4_avg_precision': precision, 't4_avg_recall': recall, 't4_avg_f1': f1, 't4_precision':precision_dict, 't4_recall': recall_dict, 't4_f1': f1_dict}

def evaluate_T5(rs_t5, gt_t5):
    actions=['Clarify', 'Answer']

    precision_dict = defaultdict(list)
    recall_dict = defaultdict(list)
    f1_dict = defaultdict(list)
    count_dict = {}
    for id in gt_t5:
        gt_action=gt_t5[id]['action']
        if id not in rs_t5:
            continue
        if gt_action[0] in actions and len(gt_t5[id]['selected_passage'])>0:
            correct=len(set(gt_t5[id]['selected_passage']).intersection(rs_t5[id]['selected_passage']))
            rs=len(rs_t5[id]['selected_passage'])
            gt=len(gt_t5[id]['selected_passage'])
            p=0
            r=0
            gt_action='-'.join(gt_action)
            if rs > 0:
                p = float(correct)/rs
                precision_dict[gt_action].append(p)
            if gt > 0:
                r = float(correct)/gt
                recall_dict[gt_action].append(r)
            f = 2. * (p * r / (p + r + min_num))
            f1_dict[gt_action].append(f)
            if gt_action not in count_dict:
                count_dict[gt_action] = 1
            else:
                count_dict[gt_action]  += 1
    precision_dict = dict(precision_dict)
    recall_dict = dict(recall_dict)
    f1_dict = dict(f1_dict)
    for gt_action in precision_dict:
        precision_dict[gt_action] = np.mean(precision_dict[gt_action])
    for gt_action in recall_dict:
        recall_dict[gt_action] = np.mean(recall_dict[gt_action])
    for gt_action in f1_dict:
        f1_dict[gt_action] = np.mean(f1_dict[gt_action])

    precision = 0
    recall = 0
    f1 = 0
    for action in precision_dict:
        precision += precision_dict[action]
    precision = precision / float(len(precision_dict) + 1)
    for action in recall_dict:
        recall += recall_dict[action]
    recall = recall / float(len(recall_dict) + 1)
    for action in f1_dict:
        f1 += f1_dict[action]
    f1 = f1 / float(len(f1_dict) + 1)
    return {'t5_avg_precision': precision, 't5_avg_recall': recall, 't5_avg_f1': f1, 't5_precision':precision_dict, 't5_recall': recall_dict, 't5_f1': f1_dict}

def evaluate_T6(rs_t6, gt_t6, tokenizer):
    actions = ['Clarify', 'Answer']

    bleu_dict={}
    rouge_dict={}
    count_dict={}
    for id in gt_t6:
        gt_action=gt_t6[id]['action']
        if gt_action[0] in actions:
            if id not in rs_t6:
                continue
            rs=rs_t6[id]['response']
            gt=tokenizer(gt_t6[id]['response'])
            gt_action='-'.join(gt_action)

            b= sentence_bleu([gt], rs, weights=(1.,), smoothing_function=SmoothingFunction().method2)
            r= sentence_rouge([gt], rs)
            if gt_action not in bleu_dict:
                bleu_dict[gt_action] = b
            else:
                bleu_dict[gt_action] += b
            if gt_action not in rouge_dict:
                rouge_dict[gt_action] = r
            else:
                rouge_dict[gt_action] += r
            if gt_action not in count_dict:
                count_dict[gt_action] = 1
            else:
                count_dict[gt_action] += 1
    for gt_action in bleu_dict:
        bleu_dict[gt_action] /= count_dict[gt_action]
        rouge_dict[gt_action] /= count_dict[gt_action]
    bleu=0
    rouge=0
    for action in bleu_dict:
        bleu += bleu_dict[action]
    bleu = bleu / float(len(bleu_dict) + 1)
    for action in rouge_dict:
        rouge += rouge_dict[action]
    rouge = rouge / float(len(rouge_dict) + 1)

    return {'t6_avg_bleu_1': bleu, 't6_avg_rouge_l': rouge, 't6_bleu_1': bleu_dict, 't6_rouge_l': rouge_dict}

def evaluate(rs_files, gt_files, tokenizer):
    rs={}
    for file in rs_files:
        with codecs.open(file, encoding='utf-8') as f:
            for line in f:
                conv = json.loads(line)
                for i in range(len(conv)):
                    if conv[-i-1]['role']=='user':
                        intent = conv[-i-1]['intent']
                        break
                selected_query = conv[-1]['selected_query']
                selected_passage = conv[-1]['selected_passage']
                rs[conv[-1]['msg_id']]={'msg_id': conv[-1]['msg_id'], 'intent':intent, 'state':conv[-1]['state'], 'action':conv[-1]['action'], 'selected_query':selected_query, 'selected_passage':selected_passage, 'response':conv[-1]['response']}

    gt={}
    for file in gt_files:
        with codecs.open(file, encoding='utf-8') as f:
            for line in f:
                conv = json.loads(line)
                for i in range(len(conv)):
                    if conv[-i-1]['role']=='user':
                        intent = conv[-i-1]['intent']
                        break
                selected_query = conv[-1]['selected_query']
                selected_passage = conv[-1]['selected_passage']
                gt[conv[-1]['msg_id']]={'msg_id': conv[-1]['msg_id'], 'intent':intent, 'state':conv[-1]['state'], 'action':conv[-1]['action'], 'selected_query':selected_query, 'selected_passage':selected_passage, 'response':conv[-1]['response']}

    t1=evaluate_T1(rs, gt)
    t2=evaluate_T2(rs, gt, tokenizer)
    t3=evaluate_T3(rs, gt)
    t4=evaluate_T4(rs, gt)
    t5=evaluate_T5(rs, gt)
    t6=evaluate_T6(rs, gt, tokenizer)
    return {**t1, **t2, **t3, **t4, **t5, **t6}