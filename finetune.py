import math, pickle
import numpy as np
import os.path as path 
from utils import util_methods
import torch, torch.nn as nn, torch.nn.functional as F
from pdb import set_trace as stop
from tqdm import tqdm

def finetune(WindowModel,ChromeModel,chrom_feature_dict,crit,optimizer,epoch,data_dict,opt,split):
    if split == 'train':
        ChromeModel.train()
    else:
        ChromeModel.eval()
    
    all_preds = torch.Tensor().cpu()
    all_targets = torch.Tensor().cpu()

    total_loss = 0
    
    if opt.adj_type in ['hic','both']:
        graph_file = path.join(opt.graph_root,split+'_graphs_'+opt.hicsize+'_'+opt.hicnorm+'norm.pkl')
        print(graph_file)
        split_adj_dict = pickle.load(open(graph_file, "rb" ))
    else:
        split_adj_dict = None
    
    chrom_count = len(chrom_feature_dict)

    for chrom in tqdm(chrom_feature_dict, mininterval=0.5,desc='('+split+')', leave=False):
        x_f = chrom_feature_dict[chrom]['forward'].cuda()
        x_r = chrom_feature_dict[chrom]['backward'].cuda()
        targets = chrom_feature_dict[chrom]['target'].cuda()
        x_f.requires_grad = True
        x_r.requires_grad = True

        split_adj = util_methods.process_graph(opt.adj_type,split_adj_dict,x_f.size(0),chrom).cuda() 

        if split=='train':
            optimizer.zero_grad()

        _,pred_f,_,z_f = ChromeModel(x_f,split_adj,None)
        _,pred_r,_,z_r = ChromeModel(x_r,split_adj,None)
        pred = (pred_f+pred_r)/2

        loss = F.binary_cross_entropy_with_logits(pred, targets.float())

        if split=='train':
            loss.backward()
            optimizer.step()
        
        total_loss += loss.sum().item()
        all_preds  = torch.cat((all_preds,F.sigmoid(pred).cpu().data),0)
        all_targets  = torch.cat((all_targets,targets.cpu().data),0)

        ##################A Saliency or TF-TF Relationships ##########################
        # Compare to CNN Preds
        # cnn_pred_f = WindowModel.module.model.relu(x_f)
        # cnn_pred_f = WindowModel.module.model.batch_norm(cnn_pred_f.cuda())
        # cnn_pred_f = WindowModel.module.model.classifier(cnn_pred_f)
        # cnn_pred_r = WindowModel.module.model.relu(x_r)
        # cnn_pred_r = WindowModel.module.model.batch_norm(cnn_pred_r.cuda())
        # cnn_pred_r = WindowModel.module.model.classifier(cnn_pred_r)
        # cnn_pred = (cnn_pred_f+cnn_pred_r)/2
        # if chrom == 'chr8' and opt.A_saliency: stop()
        ########################################################
    
    return all_preds,all_targets,total_loss
