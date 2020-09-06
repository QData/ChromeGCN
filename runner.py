# !import code; code.interact(local=vars())
import time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from pdb import set_trace as stop
from pretrain import pretrain
from finetune import finetune
from utils.evals import compute_metrics,SaveLogger

def run_epoch(WindowModel,ChromeModel,split_data,crit, optimizer,epoch,data_dict,opt,split):
    start = time.time()
    if opt.pretrain or opt.save_feats:
        print('Pretrain')
        pred,targ,loss=pretrain(WindowModel,split_data,crit,optimizer,epoch,data_dict,opt,split)
    elif not opt.save_feats:
        print('Finetune')
        pred,targ,loss=finetune(WindowModel,ChromeModel,split_data,crit,optimizer,epoch,data_dict,opt,split)

    elapsed = ((time.time()-start)/60)
    # loss = loss/float(pred.size(0))
    print('\n({split}) elapse: {elapse:3.3f} min'.format(split=split,elapse=elapsed))
    print('Loss: {loss:3.3f}'.format(loss=loss))
    return pred,targ,loss,elapsed

def run_model(WindowModel,ChromeModel,train_data,valid_data,test_data,crit,optimizer,scheduler,opt,data_dict,logger):
    valid_metrics_sums = []

    if not opt.save_feats:
        save_logger = SaveLogger(opt.model_name)

    for epoch in range(1,opt.epochs+1):
        print('================= Epoch', epoch, '=================')
        if scheduler and opt.lr_decay2 > 0: 
            scheduler.step()
        
        train_metrics,valid_metrics = None,None
        train_loss,valid_loss,valid_metrics_sum = 0,0,0
        if not opt.load_gcn and not opt.test_only:
            ####################################### TRAIN ###############################
            train_preds,train_targs,train_loss,elpsd=run_epoch(WindowModel,ChromeModel,train_data,crit,optimizer,epoch,data_dict,opt,'train')
            train_metrics = compute_metrics(train_preds,train_targs,train_loss,opt,elpsd,data_dict,opt.cell_type)

            ####################################### VALID ##############################
            valid_preds,valid_targs,valid_loss,elpsd=run_epoch(WindowModel,ChromeModel,valid_data,crit,optimizer,epoch,data_dict,opt,'valid')
            valid_metrics = compute_metrics(valid_preds,valid_targs,valid_loss,opt,elpsd,data_dict,opt.cell_type)
            valid_metrics_sum = valid_metrics['meanAUPR']+valid_metrics['meanAUPR']+valid_metrics['meanFDR']
            valid_metrics_sums += [valid_metrics_sum]

        ##################################### TEST ######################################
        test_preds,test_targs,test_loss,elpsd = run_epoch(WindowModel,ChromeModel,test_data,crit,optimizer,epoch,data_dict,opt,'test')
        test_metrics = compute_metrics(test_preds,test_targs,test_loss,opt,elpsd,data_dict,opt.cell_type)

        ################################ LOGGING #################################
        best_valid,best_test = logger.evaluate(train_metrics,valid_metrics,test_metrics,epoch,opt.total_num_parameters)
        
        if not opt.save_feats:
            save_logger.save(epoch,opt,WindowModel,ChromeModel,valid_loss,valid_metrics_sum,valid_metrics_sums,valid_preds,valid_targs,test_preds,test_targs)
            save_logger.log('test.log',epoch,test_loss,test_metrics)
            save_logger.log('valid.log',epoch,valid_loss,valid_metrics)
            save_logger.log('train.log',epoch,train_loss,train_metrics)
        
        print('best loss epoch: '+str(save_logger.best_loss_epoch)) 
        print(opt.model_name)