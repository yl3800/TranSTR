# from torch.nn.modules.module import _IncompatibleKeys
import torch
from utils.util import EarlyStopping, save_file, set_gpu_devices, pause, set_seed
import os
from utils.logger import logger
import time
import logging
import argparse
import os.path as osp
import numpy as np

parser = argparse.ArgumentParser(description="train parameter")
# general
parser.add_argument("-v", type=str, required=True, help="version")
parser.add_argument("-bs", type=int, action="store", help="BATCH_SIZE", default=128)
parser.add_argument("-lr", type=float, action="store", help="learning rate", default=1e-4)
parser.add_argument("-epoch", type=int, action="store", help="epoch for train", default=10)
parser.add_argument("-gpu", type=int, help="set gpu id", default=0)    
parser.add_argument("-es", action="store_true", help="early_stopping")
parser.add_argument("-dropout", "-drop", type=float, help="dropout rate", default=0.1)
parser.add_argument("-encoder_dropout", "-ep", type=float, help="dropout rate", default=0.1)   
parser.add_argument("-patience", "-pa", type=int, help="patience of ReduceonPleatu", default=1)
parser.add_argument("-mile_stone", "-mile", type=str, help="mile stone of MutiStepLr", default='5') 
parser.add_argument("-gamma", "-ga", type=float, help="gamma of MultiStepLR", default=0.1)
parser.add_argument("-decay", type=float, help="weight decay", default=0.001) 

# dataset
parser.add_argument('-dataset', default='msrvtt-qa',choices=['msrvtt-qa', 'msvd-qa','next-qa'], type=str)
parser.add_argument("-objs", default=10, type=int, help="sample of object feature")

# model
parser.add_argument("-d_model", "-md",  type=int, help="hidden dim of vq encoder", default=768) 
parser.add_argument("-word_dim", "-wd", type=int, help="word dim ", default=768)   
parser.add_argument("-topK_frame", "-fk", type=int, help="word dim ", default=8)   
parser.add_argument("-topK_obj", "-ok", type=int, help="word dim ", default=5)   
parser.add_argument("-hard_eval", "-hd", action="store_true", help="hard selection during inference")

# transformer
parser.add_argument("-num_encoder_layers", "-el", type=int, help="number of encoder layers in transformer", default=1)
parser.add_argument("-num_decoder_layers", "-dl", type=int, help="number of decoder layers in transformer", default=1)
parser.add_argument("-n_query", type=int, help="num of query", default=5) 
parser.add_argument("-nheads", type=int, help="num of attention head", default=8) 
parser.add_argument("-normalize_before", action="store_true", help="pre or post normalize")
parser.add_argument("-activation", default='relu', choices=['relu','gelu','glu'], type=str)

# lan model
parser.add_argument("-text_encoder_lr","-tlr", type=float, action="store", help="learning rate for lan model", default=1e-5)
parser.add_argument("-freeze_text_encoder", action="store_true", help="freeze text encoder")
parser.add_argument("-text_encoder_type", "-t", default="microsoft/deberta-base", choices=["roberta-base","distilroberta-base",\
                    "bert-base-uncased", "distilbert-base-uncased","microsoft/deberta-base",\
                        "microsoft/deberta-v3-base","microsoft/deberta-v3-small", "microsoft/deberta-v3-xsmall"], type=str)

args = parser.parse_args()
set_gpu_devices(args.gpu)
set_seed(999)
set_gpu_devices(args.gpu)

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, MultiStepLR
from networks.model import VideoQAmodel
# from dataloader.dataset import VidQADataset 
from DataLoader import VideoQADataset

# from torch.utils.tensorboard import SummaryWriter
torch.set_printoptions(linewidth=200)
np.set_printoptions(edgeitems=30, linewidth=30, formatter=dict(float=lambda x: "%.3g" % x))
# torch.autograd.set_detect_anomaly(True)


def train(model, optimizer, train_loader, xe, device):
    model.train()
    total_step = len(train_loader)
    epoch_loss = 0.0
    prediction_list = []
    answer_list = []
    for iter, inputs in enumerate(train_loader):
        # videos, qns_w, ans_w, ans_id, _ = inputs
        # video_inputs = videos.to(device)
        vid_frame_inputs, vid_obj_inputs, qns_w, ans_id, _ = inputs
        vid_frame_feat = vid_frame_inputs.to(device)
        vid_obj_feat = vid_obj_inputs.to(device)
        ans_targets = ans_id.to(device)
        with torch.cuda.amp.autocast(enabled=True):
            out_f = model( vid_frame_feat, vid_obj_feat, qns_w)
            loss = xe(out_f, ans_targets)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        epoch_loss += loss.item()
        prediction=out_f.max(-1)[1] # bs,
        prediction_list.append(prediction)
        answer_list.append(ans_id)

    predict_answers = torch.cat(prediction_list, dim=0).long().cpu()
    ref_answers = torch.cat(answer_list, dim=0).long()
    acc_num = torch.sum(predict_answers==ref_answers).numpy()
    
    return epoch_loss / total_step, acc_num*100.0 / len(ref_answers)
    

def eval(model, val_loader, device):
    model.eval()
    prediction_list = []
    answer_list = []
    with torch.no_grad():
        for iter, inputs in enumerate(val_loader):
            # videos, qns_w, ans_w, ans_id, _ = inputs
            # video_inputs = videos.to(device)
            vid_frame_inputs, vid_obj_inputs, qns_w, ans_id, _ = inputs
            vid_frame_feat = vid_frame_inputs.to(device)
            vid_obj_feat = vid_obj_inputs.to(device)
            out = model( vid_frame_feat, vid_obj_feat, qns_w)
            prediction=out.max(-1)[1] # bs,            
            prediction_list.append(prediction)
            answer_list.append(ans_id)

    predict_answers = torch.cat(prediction_list, dim=0).long().cpu()
    ref_answers = torch.cat(answer_list, dim=0).long()
    acc_num = torch.sum(predict_answers==ref_answers).numpy()
    return acc_num*100.0 / len(ref_answers)


def predict(model,test_loader, device):
    """
    predict the answer with the trained model
    :param model_file:
    :return:
    """

    model.eval()
    results = {}
    prediction_list = []
    answer_list = []
    with torch.no_grad():
        for iter, inputs in enumerate(test_loader):
            # videos, qns_w, ans_w, ans_id, qns_keys = inputs
            # video_inputs = videos.to(device)
            vid_frame_inputs, vid_obj_inputs, qns_w, ans_id, qns_keys = inputs
            vid_frame_feat = vid_frame_inputs.to(device)
            vid_obj_feat = vid_obj_inputs.to(device)
            out = model(vid_frame_feat, vid_obj_feat, qns_w)
            prediction=out.max(-1)[1] # bs,
            prediction_list.append(prediction)
            answer_list.append(ans_id)

            for qid, pred, ans in zip(qns_keys, prediction.data.cpu().numpy(), ans_id.numpy()):
                results[qid] = {'prediction': int(pred), 'answer': int(ans)}
    
    predict_answers = torch.cat(prediction_list, dim=0).long().cpu()
    ref_answers = torch.cat(answer_list, dim=0).long()
    acc_num = torch.sum(predict_answers==ref_answers).numpy()

    return results, acc_num*100.0 / len(ref_answers)


if __name__ == "__main__":

    # writer = SummaryWriter('./log/tensorboard')
    logger, sign =logger(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset=VideoQADataset('train', args.objs)
    val_dataset=VideoQADataset('val', args.objs)
    test_dataset=VideoQADataset('test',  args.objs)

    train_loader = DataLoader(dataset=train_dataset,batch_size=args.bs,shuffle=True,num_workers=8,pin_memory=True, prefetch_factor=4)
    val_loader = DataLoader(dataset=val_dataset,batch_size=args.bs,shuffle=False,num_workers=8,pin_memory=True, prefetch_factor=4)
    test_loader = DataLoader(dataset=test_dataset,batch_size=args.bs,shuffle=False,num_workers=8,pin_memory=True, prefetch_factor=4)

    # hyper setting
    epoch_num = args.epoch
    args.device = device
    config = {**vars(args)}
    model = VideoQAmodel(**config)
    param_dicts = [
    {"params": [p for n, p in model.named_parameters() if "text_encoder" not in n and p.requires_grad]},
    {"params": [p for n, p in model.named_parameters() if "text_encoder" in n and p.requires_grad], "lr": args.text_encoder_lr}]
    optimizer = torch.optim.AdamW(params = param_dicts, lr=args.lr, weight_decay=args.decay)
    # optimizer = torch.optim.AdamW(params = [{'params':model.parameters()}], lr=args.lr, weight_decay=args.decay)
    # scheduler = ReduceLROnPlateau(optimizer, 'max', factor=args.gamma, patience=args.patience, verbose=True)
    # scheduler = StepLR(optimizer, step_size=4, gamma=0.5)
    scheduler = MultiStepLR(optimizer, milestones=[int(item) for item in args.mile_stone.split(',')], gamma=args.gamma, verbose=True)
    model.to(device)
    xe = nn.CrossEntropyLoss().to(device)
    # Creates a GradScaler once at the beginning of training.
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    # train & val
    best_eval_score = 0.0
    best_epoch=1
    for epoch in range(1, epoch_num+1):
        train_loss, train_acc = train(model, optimizer, train_loader, xe, device)
        eval_score = eval(model, val_loader, device)
        scheduler.step()
        if eval_score > best_eval_score :
            best_eval_score = eval_score
            best_epoch = epoch 
            best_model_path='./models/best_model-{}.ckpt'.format(sign)
            torch.save(model.state_dict(), best_model_path)
        
        test_score = eval(model, test_loader, device)
        logger.debug("==>Epoch:[{}/{}][LR{}][Train Loss: {:.4f} Train acc: {:.2f} Val acc: {:.2f} Test: {:.2f}".
        format(epoch, epoch_num, optimizer.param_groups[0]['lr'], train_loss,train_acc, eval_score, test_score))

    logger.debug("Epoch {} Best Val acc{:.2f}".format(best_epoch, best_eval_score))

    # predict with best model
    model.load_state_dict(torch.load(best_model_path))
    results, test_acc=predict(model,test_loader, device)
    logger.debug("Test acc{:.2f} on {} epoch".format(test_acc, best_epoch))

    result_path= './prediction/{}-{}-{:.2f}.json'.format(sign, best_epoch, best_eval_score)
    save_file(results, result_path)
