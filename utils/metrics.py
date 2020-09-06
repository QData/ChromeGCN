

import numpy
import scipy.sparse as sp
import logging
from collections import OrderedDict
import sys
import pdb
from sklearn import metrics as skmetrics
from threading import Lock
from threading import Thread
import torch
import math
from pdb import set_trace as stop
import os
import numpy as np
import pandas as pd


FORMAT = '[%(asctime)s] %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
LOGGER = logging.getLogger(__name__)


def mean_average_precision(true_targets, predictions):
    return skmetrics.average_precision_score(true_targets, predictions, average='macro', pos_label=1)


def subset_accuracy(true_targets, predictions, per_sample=False, axis=0):
    result = numpy.all(true_targets == predictions, axis=axis)
    if not per_sample:
        result = numpy.mean(result)
    return result


def hamming_loss(true_targets, predictions, per_sample=False, axis=0):
    result = numpy.mean(numpy.logical_xor(true_targets, predictions),
                        axis=axis)
    if not per_sample:
        result = numpy.mean(result)
    return result


def tp_fp_fn(true_targets, predictions, axis=0):
    # axis: axis for instance
    tp = numpy.sum(true_targets * predictions, axis=axis).astype('float32')
    fp = numpy.sum(numpy.logical_not(true_targets) * predictions,
                   axis=axis).astype('float32')
    fn = numpy.sum(true_targets * numpy.logical_not(predictions),
                   axis=axis).astype('float32')
    return (tp, fp, fn)


def example_f1_score(true_targets, predictions, per_sample=False, axis=0):
    tp, fp, fn = tp_fp_fn(true_targets, predictions, axis=axis)
    numerator = 2*tp
    denominator = (numpy.sum(true_targets,axis=axis).astype('float32') + numpy.sum(predictions,axis=axis).astype('float32'))
    zeros = numpy.where(denominator == 0)[0]
    denominator = numpy.delete(denominator,zeros)
    numerator = numpy.delete(numerator,zeros)
    example_f1 = numerator/denominator

    if per_sample:
        f1 = example_f1
    else:
        f1 = numpy.mean(example_f1)

    return f1



def f1_score_from_stats(tp, fp, fn, average='micro'):
    assert len(tp) == len(fp)
    assert len(fp) == len(fn)

    if average not in set(['micro', 'macro']):
        raise ValueError("Specify micro or macro")

    if average == 'micro':
        f1 = 2*numpy.sum(tp) / \
            float(2*numpy.sum(tp) + numpy.sum(fp) + numpy.sum(fn))

    elif average == 'macro':

        def safe_div(a, b):
            """ ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0] """
            with numpy.errstate(divide='ignore', invalid='ignore'):
                c = numpy.true_divide(a, b)
            return c[numpy.isfinite(c)]

        f1 = numpy.mean(safe_div(2*tp, 2*tp + fp + fn))

    return f1


def f1_score(true_targets, predictions, average='micro', axis=0):
    """
        average: str
            'micro' or 'macro'
        axis: 0 or 1
            label axis
    """
    if average not in set(['micro', 'macro']):
        raise ValueError("Specify micro or macro")

    tp, fp, fn = tp_fp_fn(true_targets, predictions, axis=axis)
    f1 = f1_score_from_stats(tp, fp, fn, average=average)

    return f1


def aupr_threaded(all_targets,all_predictions):
    
    aupr_array = []
    lock = Lock()

    def aupr_(start,end,all_targets,all_predictions):
        for i in range(all_targets.shape[1]):
            try:
                precision, recall, thresholds = skmetrics.precision_recall_curve(all_targets[:,i], all_predictions[:,i], pos_label=1)
                auPR = skmetrics.auc(recall,precision)
                lock.acquire() 
                aupr_array.append(numpy.nan_to_num(auPR))
                lock.release()
            except Exception: 
                pass
                 
    t1 = Thread(target=aupr_, args=(0,100,all_targets,all_predictions) )
    t2 = Thread(target=aupr_, args=(100,200,all_targets,all_predictions) )
    t3 = Thread(target=aupr_, args=(200,300,all_targets,all_predictions) )
    t4 = Thread(target=aupr_, args=(300,400,all_targets,all_predictions) )
    t5 = Thread(target=aupr_, args=(400,500,all_targets,all_predictions) )
    t6 = Thread(target=aupr_, args=(500,600,all_targets,all_predictions) )
    t7 = Thread(target=aupr_, args=(600,700,all_targets,all_predictions) )
    t8 = Thread(target=aupr_, args=(700,800,all_targets,all_predictions) )
    t9 = Thread(target=aupr_, args=(800,900,all_targets,all_predictions) )
    t10 = Thread(target=aupr_, args=(900,919,all_targets,all_predictions) )
    t1.start();t2.start();t3.start();t4.start();t5.start();t6.start();t7.start();t8.start();t9.start();t10.start()
    t1.join();t2.join();t3.join();t4.join();t5.join();t6.join();t7.join();t8.join();t9.join();t10.join()
    

    aupr_array = numpy.array(aupr_array)

    mean_aupr = numpy.mean(aupr_array)
    median_aupr = numpy.median(aupr_array)
    return mean_aupr,median_aupr,aupr_array

def fdr(all_targets,all_predictions, fdr_cutoff=0.5):
    fdr_array = []
    for i in range(all_targets.shape[1]):
        try:
            precision, recall, thresholds = skmetrics.precision_recall_curve(all_targets[:,i], all_predictions[:,i],pos_label=1)
            fdr = 1- precision
            cutoff_index = next(i for i, x in enumerate(fdr) if x <= fdr_cutoff)
            fdr_at_cutoff = recall[cutoff_index]
            if not math.isnan(fdr_at_cutoff):
                fdr_array.append(numpy.nan_to_num(fdr_at_cutoff))
        except: 
            pass
    
    fdr_array = numpy.array(fdr_array)
    mean_fdr = numpy.mean(fdr_array)
    median_fdr = numpy.median(fdr_array)
    var_fdr = numpy.var(fdr_array)
    return mean_fdr,median_fdr,var_fdr,fdr_array


def aupr(all_targets,all_predictions):
    aupr_array = []
    for i in range(all_targets.shape[1]):
        try:
            precision, recall, thresholds = skmetrics.precision_recall_curve(all_targets[:,i], all_predictions[:,i], pos_label=1)
            auPR = skmetrics.auc(recall,precision)
            if not math.isnan(auPR):
                aupr_array.append(numpy.nan_to_num(auPR))
        except: 
            pass
    
    aupr_array = numpy.array(aupr_array)
    mean_aupr = numpy.mean(aupr_array)
    median_aupr = numpy.median(aupr_array)
    var_aupr = numpy.var(aupr_array)
    return mean_aupr,median_aupr,var_aupr,aupr_array



def auroc_threaded(all_targets,all_predictions):
    
    auc_array = []
    lock = Lock()

    def auroc_(start,end,all_targets,all_predictions):
        for i in range(start,end):
            try:  
                auROC = skmetrics.roc_auc_score(all_targets[:,i], all_predictions[:,i])
                lock.acquire() 
                if not math.isnan(auROC):
                    auc_array.append(auROC)
                lock.release()
            except ValueError:
                pass
                
    t1 = Thread(target=auroc_, args=(0,100,all_targets,all_predictions) )
    t2 = Thread(target=auroc_, args=(100,200,all_targets,all_predictions) )
    t3 = Thread(target=auroc_, args=(200,300,all_targets,all_predictions) )
    t4 = Thread(target=auroc_, args=(300,400,all_targets,all_predictions) )
    t5 = Thread(target=auroc_, args=(400,500,all_targets,all_predictions) )
    t6 = Thread(target=auroc_, args=(500,600,all_targets,all_predictions) )
    t7 = Thread(target=auroc_, args=(600,700,all_targets,all_predictions) )
    t8 = Thread(target=auroc_, args=(700,800,all_targets,all_predictions) )
    t9 = Thread(target=auroc_, args=(800,900,all_targets,all_predictions) )
    t10 = Thread(target=auroc_, args=(900,919,all_targets,all_predictions) )
    t1.start();t2.start();t3.start();t4.start();t5.start();t6.start();t7.start();t8.start();t9.start();t10.start()
    t1.join();t2.join();t3.join();t4.join();t5.join();t6.join();t7.join();t8.join();t9.join();t10.join()
    
    auc_array = numpy.array(auc_array)

    mean_auc = numpy.mean(auc_array)
    median_auc = numpy.median(auc_array)
    return mean_auc,median_auc,auc_array



def Find_Optimal_Cutoff(all_targets, all_predictions):
    thresh_array = []
    for j in range(all_targets.shape[1]):
        try:
            fpr, tpr, threshold = skmetrics.roc_curve(all_targets[:,j], all_predictions[:,j], pos_label=1)
            i = numpy.arange(len(tpr)) 
            roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
            roc_t = roc.ix[(roc.tf-0).abs().argsort()[:1]]
            thresh_array.append(list(roc_t['threshold'])[0])
        except: 
            pass
    return thresh_array


def auroc(all_targets,all_predictions):
    auc_array = []
    lock = Lock()
    
    for i in range(all_targets.shape[1]):
        try:  
            auROC = skmetrics.roc_auc_score(all_targets[:,i], all_predictions[:,i])
            auc_array.append(auROC)
        except ValueError:
            pass
    
    auc_array = numpy.array(auc_array)
    mean_auc = numpy.mean(auc_array)
    median_auc = numpy.median(auc_array)
    var_auc = numpy.var(auc_array)
    return mean_auc,median_auc,var_auc,auc_array

def plot_auroc(all_targets_in,all_predictions_in,name1='',name2=''):
    # stop()
    auc_array = []
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    for i in range(all_targets_in.shape[1]):
        try:  
            fpr, tpr, thresholds = skmetrics.roc_curve(all_targets_in[:,i], all_predictions_in[:,i], pos_label=1)
            ax.plot(fpr, tpr)#,color='k')
            auc = skmetrics.auc(fpr, tpr)
            auc_array.append(auc)
        except ValueError:
            pass
    auc_array = np.array(auc_array)
    mu=auc_array.mean()
    median = np.median(auc_array)
    sigma = auc_array.std()
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    textstr = '\n'.join(( r'$\mu=%.3f$' % (mu, ), r'$\mathrm{median}=%.3f$' % (median, ), r'$\sigma=%.3f$' % (sigma, )))
    ax.text(0.64, 0.23, textstr, transform=ax.transAxes, fontsize=15,verticalalignment='top', bbox=props)
    fig.suptitle(name1, fontsize=18,y=0.95)
    plt.xlabel('FPR', fontsize=15)
    plt.ylabel('TPR', fontsize=15)
    fig.savefig('plots/'+name2+'_auc.png',bbox_inches='tight')

def plot_aupr(all_targets_in,all_predictions_in,name1='',name2=''):
    # stop()
    aupr_array = []
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    for i in range(all_targets_in.shape[1]):
        try:  
            precision, recall, thresholds = skmetrics.precision_recall_curve(all_targets_in[:,i], all_predictions_in[:,i], pos_label=1)
            ax.plot(recall,precision)#,color='k')
            auPR = skmetrics.auc(recall,precision)
            aupr_array.append(auPR)
        except ValueError:
            pass
    aupr_array = np.array(aupr_array)
    mu=aupr_array.mean()
    median = np.median(aupr_array)
    sigma = aupr_array.std()
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    textstr = '\n'.join(( r'$\mu=%.3f$' % (mu, ), r'$\mathrm{median}=%.3f$' % (median, ), r'$\sigma=%.3f$' % (sigma, )))
    ax.text(0.64, 0.95, textstr, transform=ax.transAxes, fontsize=15,verticalalignment='top', bbox=props)
    fig.suptitle(name1, fontsize=18,y=0.95)
    plt.xlabel('Recall', fontsize=15)
    plt.ylabel('Precision', fontsize=15)
    fig.savefig('plots/'+name2+'_aupr.png',bbox_inches='tight')