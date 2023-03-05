from operator import gt
import os.path as osp
from unittest import result
from utils.util import load_file
import argparse
import logging
logger = logging.getLogger('VQA') 

map_name = {'d': 'Des   ', 'e': 'Exp   ', 'p': 'Pred-A', 'c': 'CF-A  ', 'pr': 'Pred-R', 'cr': 'CF-R  ', 'par':'Pred  ', 'car': 'CF    ', 'all': 'ALL   '}

def accuracy_metric(result_file, qtype):
    if qtype == -1:
        accuracy_metric_cvid(result_file)
    if qtype == 0:
        accuracy_metric_q0(result_file)
    if qtype == 1:
        accuracy_metric_q1(result_file)
    if qtype == 2:
        accuracy_metric_q2(result_file)
    if qtype == 3:
        accuracy_metric_q3(result_file)

def accuracy_metric_q0(result_file):
    preds = list(load_file(result_file).items())
    group_acc = {'D': 0}
    group_cnt = {'D': 0}
    all_acc = 0
    all_cnt = 0
    for idx in range(len(preds)):
        id_qtypes = preds[idx]
        answer = id_qtypes[1]['answer']
        pred = id_qtypes[1]['prediction']
        group_cnt['D'] += 1
        all_cnt += 1
        if answer == pred:
            group_acc['D'] += 1
            all_acc += 1
    for qtype, acc in group_acc.items(): #
        print('{0:21} ==> {1:6.2f}%'.format(map_name[qtype], acc*100.0/group_cnt[qtype]))
    print('{0:21} ==> {1:6.2f}%'.format('Acc', all_acc*100.0/all_cnt))

def accuracy_metric_q1(result_file):
    preds = list(load_file(result_file).items())
    group_acc = {'E': 0}
    group_cnt = {'E': 0}
    all_acc = 0
    all_cnt = 0
    for idx in range(len(preds)):
        id_qtypes = preds[idx]
        answer = id_qtypes[1]['answer']
        pred = id_qtypes[1]['prediction']
        group_cnt['E'] += 1
        all_cnt += 1
        if answer == pred:
            group_acc['E'] += 1
            all_acc += 1
    for qtype, acc in group_acc.items(): #
        print('{0:21} ==> {1:6.2f}%'.format(map_name[qtype], acc*100.0/group_cnt[qtype]))
    print('{0:21} ==> {1:6.2f}%'.format('Acc', all_acc*100.0/all_cnt))

def accuracy_metric_q2(result_file):
    preds = list(load_file(result_file).items())
    qtype2short = ['PA', 'PR', 'P']
    group_acc = {'PA': 0, 'PR': 0, 'P': 0}
    group_cnt = {'PA': 0, 'PR': 0, 'P': 0}
    all_acc = 0
    all_cnt = 0
    for idx in range(len(preds)//2):
        id_qtypes = preds[idx*2:(idx+1)*2]
        qtypes = [0, 1]
        answer = [ans_pre[1]['answer'] for ans_pre in id_qtypes]
        pred = [ans_pre[1]['prediction'] for ans_pre in id_qtypes]
        for i in range(2):
            group_cnt[qtype2short[qtypes[i]]] += 1
            if answer[i] == pred[i]:
                group_acc[qtype2short[qtypes[i]]] += 1
        group_cnt['P'] += 1
        all_cnt += 1
        if answer[0] == pred[0] and answer[1] == pred[1]:
            group_acc['P'] += 1
            all_acc += 1
    for qtype, acc in group_acc.items(): #
        print('{0:21} ==> {1:6.2f}%'.format(map_name[qtype], acc*100.0/group_cnt[qtype]))
    print('{0:21} ==> {1:6.2f}%'.format('Acc', all_acc*100.0/all_cnt))

def accuracy_metric_q3(result_file):
    preds = list(load_file(result_file).items())
    qtype2short = ['CA', 'CR', 'C']
    group_acc = {'CA': 0, 'CR': 0, 'C': 0}
    group_cnt = {'CA': 0, 'CR': 0, 'C': 0}
    all_acc = 0
    all_cnt = 0
    for idx in range(len(preds)//2):
        id_qtypes = preds[idx*2:(idx+1)*2]
        qtypes = [0, 1]
        answer = [ans_pre[1]['answer'] for ans_pre in id_qtypes]
        pred = [ans_pre[1]['prediction'] for ans_pre in id_qtypes]
        for i in range(2):
            group_cnt[qtype2short[qtypes[i]]] += 1
            if answer[i] == pred[i]:
                group_acc[qtype2short[qtypes[i]]] += 1
        group_cnt['C'] += 1
        all_cnt += 1
        if answer[0] == pred[0] and answer[1] == pred[1]:
            group_acc['C'] += 1
            all_acc += 1
    for qtype, acc in group_acc.items(): #
        print('{0:21} ==> {1:6.2f}%'.format(map_name[qtype], acc*100.0/group_cnt[qtype]))
    print('{0:21} ==> {1:6.2f}%'.format('Acc', all_acc*100.0/all_cnt))

def accuracy_metric_all(result_file):
    preds = list(load_file(result_file).items())
    qtype2short = ['D', 'E', 'PA', 'PR', 'CA', 'CR', 'P', 'C']
    group_acc = {'D': 0, 'E': 0, 'PA': 0, 'PR': 0, 'CA': 0, 'CR': 0, 'P': 0, 'C': 0}
    group_cnt = {'D': 0, 'E': 0, 'PA': 0, 'PR': 0, 'CA': 0, 'CR': 0, 'P': 0, 'C': 0}
    all_acc = 0
    all_cnt = 0
    for idx in range(len(preds)//6):
        id_qtypes = preds[idx*6:(idx+1)*6]
        qtypes = [int(id_qtype[0].split('_')[-1]) for id_qtype in id_qtypes]
        answer = [ans_pre[1]['answer'] for ans_pre in id_qtypes]
        pred = [ans_pre[1]['prediction'] for ans_pre in id_qtypes]
        for i in range(6):
            group_cnt[qtype2short[qtypes[i]]] += 1
            if answer[i] == pred[i]:
                group_acc[qtype2short[qtypes[i]]] += 1
        group_cnt['C'] += 1
        group_cnt['P'] += 1
        all_cnt += 4
        if answer[0] == pred[0]:
            all_acc += 1
        if answer[1] == pred[1]:
            all_acc += 1
        if answer[2] == pred[2] and answer[3] == pred[3]:
            group_acc['P'] += 1
            all_acc += 1
        if answer[4] == pred[4] and answer[5] == pred[5]:
            group_acc['C'] += 1
            all_acc += 1
    for qtype, acc in group_acc.items(): #
        print('{0:21} ==> {1:6.2f}%'.format(map_name[qtype], acc*100.0/group_cnt[qtype]))
    print('{0:21} ==> {1:6.2f}%'.format('Acc', all_acc*100.0/all_cnt))


def accuracy_metric_cvid(result_file, gt_file):
    
    qtypes = ['d', 'e', 'p', 'pr', 'c', 'cr', 'par', 'car']
    group_acc = {'d':0, 'e':0, 'p':0, 'pr':0, 'c':0, 'cr':0, 'par':0, 'car':0}
    group_cnt = {'d':0, 'e':0, 'p':0, 'pr':0, 'c':0, 'cr':0, 'par':0, 'car':0}
    all_acc = 0
    all_cnt = 0
    gts = load_file(gt_file)
    qns_group = {'d':[], 'e':[], 'p':[], 'pr':[], 'c':[], 'cr':[]}
    for idx, row in gts.iterrows():
         vid, qtype = row['video_id'], row['type']
         qid = vid+'_'+qtype
         qns_group[qtype].append(qid)

    preds = load_file(result_file)
    acc_par = 0
    acc_car = 0
    for qtype, qids in qns_group.items():
        acc = 0
        for qid in qids:
            answer = preds[qid]['answer']
            pred = preds[qid]['prediction']
            if answer == pred: 
                acc += 1
                if qtype in ['p', 'c']:
                    # print(preds[qid+'r'])
                    if preds[qid+'r']['answer'] == preds[qid+'r']['prediction']:
                        if qtype == 'p': acc_par += 1
                        if qtype == 'c': acc_car += 1

        group_cnt[qtype] = len(qids)
        group_acc[qtype] = acc
    
    # print(acc_par, acc_car)
    group_acc['par'] = acc_par
    group_acc['car'] = acc_car
    group_cnt['par'] = group_cnt['p']
    group_cnt['car'] = group_cnt['c']

    all_acc = group_acc['d'] + group_acc['e'] + group_acc['par'] + group_acc['car']
    all_cnt = group_cnt['p'] + group_cnt['e'] + group_cnt['par'] + group_cnt['car']

    # for qtype, acc in group_acc.items(): 
    #     print(map_name[qtype], end='\t')
    # print('All')
    # for qtype, acc in group_acc.items(): 
    #     print('{:.2f}'.format(acc*100.0/group_cnt[qtype]), end='\t')
    # # print('')
    # print('{:.2f}'.format(all_acc*100.0/all_cnt))
    logger.debug("    ".join(list(map(lambda q_type: map_name[q_type], group_acc.keys()))))
    logger.debug("     ".join(['{:.2f}'.format(acc*100.0/group_cnt[qtype]) for qtype, acc in group_acc.items()]))
    logger.debug('Acc: {:.2f}'.format(all_acc*100.0/all_cnt)) 



def main(result_file, mode='val'):
    print('Evaluating {}'.format(result_file))
    dataset_dir = '../data/datasets/causalvid/'
    gt_file = osp.join(dataset_dir, mode+'.csv')
    accuracy_metric_cvid(result_file, gt_file)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--mode", type=str, default='val', choices=['val','test'])
    # parser.add_argument("--folder", type=str)
    # args = parser.parse_args()

    # result_file = f'./save_models/causalvid/{args.folder}/{args.mode}-res.json'
    # main(result_file, args.mode)
    result_file = "convert.json"  # prediction文件
    gt_file = "/storage_fast/ycli/vqa/qa_dataset/causalvid/with_qid/test.csv"  # gt_file指的就是生成这个prediction文件的数据集csv文件
    accuracy_metric_cvid(result_file, gt_file=gt_file)