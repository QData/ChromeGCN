import math
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from tqdm import tqdm
from utils import util_methods
from pdb import set_trace as stop

def pretrain(WindowModel,split_data, crit, optimizer,epoch,data_dict,opt,split):
    if opt.pretrain and split == 'train':
        WindowModel.train()
    else:
        WindowModel.eval()

    total_loss = 0
    batch_size = split_data._batch_size

    all_preds = torch.zeros(len(split_data._src_insts),opt.tgt_vocab_size).cpu()
    all_targs = torch.zeros(len(split_data._src_insts),opt.tgt_vocab_size).cpu()
    all_x_f = torch.Tensor().cpu()
    all_x_r = torch.Tensor().cpu()
    all_locs = []
    
    batch_idx = 0
    num_batches= math.floor(len(split_data._src_insts)/float(batch_size))
    pbar = tqdm(total=num_batches,mininterval=0.5,desc=split, leave=False)

    for batch in split_data:
        pbar.update()
        src,loc,tgt = batch

        if opt.pretrain and split=='train':
            optimizer.zero_grad()

        if (batch[0][0].size(0) < opt.batch_size): # multi-gpu padding
            src,tgt = util_methods.pad_batch(opt.batch_size,src,tgt)

        x_out_f,x_out_r,pred,attn_f,attn_r = WindowModel(src[0], data_dict['src'])

        if (batch[0][0].size(0) < opt.batch_size): # multi-gpu unpadding 
            x_out_f,x_out_r,pred,tgt,attn_f,attn_r = util_methods.unpad_batch(batch[0][0].size(0),x_out_f,x_out_r,pred,tgt,attn_f,attn_r)

        if attn_f is not None: attn = (attn_f + attn_r / 2)

        loss = F.binary_cross_entropy_with_logits(pred, tgt.float())

        if opt.pretrain and split=='train':
            loss.backward()
            optimizer.step()
        
        ## Updates ##
        total_loss += loss.sum().item()
        start_idx, end_idx = (batch_idx*batch_size),((batch_idx+1)*batch_size)
        all_preds[start_idx:end_idx] = F.sigmoid(pred).cpu().data
        all_targs[start_idx:end_idx] = tgt.cpu().data
        batch_idx +=1

        if opt.save_feats:
            all_x_f = torch.cat((all_x_f,x_out_f.detach().cpu()),0)
            all_x_r = torch.cat((all_x_r,x_out_r.detach().cpu()),0)
            for loc_i in loc: all_locs.append(loc_i) 

    if opt.save_feats:
        util_methods.save_feats(opt.model_name,split,all_targs,all_locs,all_x_f,all_x_r)

    pbar.close()

    return all_preds,all_targs,total_loss
