#!/usr/bin/env python
import argparse
from pdb import set_trace as stop
import os.path as path 
import sys,pickle
import csv,os
from pathlib import Path
import torch
import numpy as np
from evals.evals import compute_metrics
from scipy import stats
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import utils
from scipy import stats
import random
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()

parser.add_argument('-dataset', type=str, default='encode')
parser.add_argument('-cell_type', type=str, default='GM12878') 
parser.add_argument('-save_path', type=str, default='./outputs/') 
parser.add_argument('-viz_path', type=str, default='plots')
parser.add_argument('-hicnorm', type=str, choices=['KR', 'VC','SQRTVC'], default='SQRTVC')
parser.add_argument('-hicsize', type=str, choices=['500000','2500000','5000000'], default='500000')
parser.add_argument('-max', type=int, default=100) 
parser.add_argument('-plot_diffs', action='store_true')
parser.add_argument('-plot_comparisons', action='store_true')
parser.add_argument('-violin_plots', action='store_true')


args = parser.parse_args()

args.br_threshold = 0.5

args.viz_path = args.viz_path+'/'+args.cell_type+'/hic/'
if not os.path.exists(args.viz_path):
    os.makedirs(args.viz_path)

if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)


root = '/bigtemp/jjl5sw/deepENCODE/results/'
# rm /bigtemp/jjl5sw/deepENCODE/results/encode/K562/*/test_metrics.pt


data_dict = torch.load('/bigtemp/jjl5sw/ChromeGCN/data/GM12878/train_valid_test_small.pt')['dict']
label_dict = data_dict['tgt']

label_list = [key.replace('wgencodeawg','').replace('unipk','').replace('gm12878','').replace('k562','').replace('iggmus','').replace('sydh','').replace('uta','').replace('haib','').replace('pcr1x','').replace('pcr2x','').replace('tfbs','tfbs_').replace('iggrab','').replace('5200401194','').replace('broad','').split('sc')[0] for key in label_dict.keys()]

variations = { 
'GNN_hic':[
        'graph.expecto.128.bsz_64.loss_ce.sgd.lr_25.drop_20_20',
        'graph.expecto.128.bsz_64.loss_ce.sgd.lr_25.drop_20_20.finetune.lr2_25.gcndrop_20.sgd.gcn.layers_2.gate.adj_both.norm_SQRTVC.new',
        ],
}

# THRESHOLDS = [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.8,0.85,0.9,0.95]
THRESHOLDS = [0.50]

METRICS = ['ebF1','miF1','maF1','ACC','HA']

def plot_comparison(x,y,metric,variation,viz_path,cell_type):
    t_test_statistic,t_test_pvalue = stats.ttest_rel(x,y)
    
    fig, ax = plt.subplots()
    ax.scatter(x, y,color='#ff0055')
    lims = [np.min([ax.get_xlim(), ax.get_ylim()]), np.max([ax.get_xlim(), ax.get_ylim()])]
    ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    
    if metric == 'Recall_at_50':
        ax.set_xlabel('CNN Recall at 50% FDR')
        # ax.set_ylabel(variation+' '+metric)
        ax.set_ylabel('GNN (Hi-C) Recall at 50% FDR')
    else:
        ax.set_xlabel('CNN '+metric.replace('_',' '))
        # ax.set_ylabel(variation+' '+metric)
        ax.set_ylabel('GNN (Hi-C) '+metric.replace('_',' '))
    
    ax.set_title(cell_type)


    textstr = '\n'.join(( r't-statistic=%.2f' % (t_test_statistic, ), r'pvalue=%.2E' % (t_test_pvalue, )))
    props = dict(boxstyle='round', facecolor='#f5f5f5', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=13, verticalalignment='top', bbox=props)

    plt.savefig(viz_path+cell_type+'_'+variation.replace(' ','_')+'_'+metric+'_comparison.pdf',dpi=500,bbox_inches = 'tight', pad_inches = 0.3)

def plot_label_difference(cnn,gnn,metric,variation,viz_path,cell_type,norm_weights=None):
    # diff = (gnn-cnn)/gnn
    diff = (gnn-cnn)
    # if metric == 'AUPR':
    #     stop()
    
    rank_indices_y = np.argsort(norm_weights.numpy())
    diff = diff[rank_indices_y]
    
    degrees = np.sort(norm_weights.numpy())#.astype(int)
    fig, ax = plt.subplots()
    
    ax.axhline(linewidth=1, color='k')

    t_test_statistic,t_test_pvalue = stats.ttest_rel(cnn,gnn)
    textstr = '\n'.join(( r't-statistic=%.2f' % (t_test_statistic, ), r'pvalue=%.2E' % (t_test_pvalue, )))
    props = dict(boxstyle='round', facecolor='#f5f5f5', alpha=0.4)
    ax.text(0.03, 0.97, textstr, transform=ax.transAxes, fontsize=13, verticalalignment='top', bbox=props)

    neg_mask = diff < 0
    pos_mask = diff >=0


    sorted_labels = np.array(label_list)[rank_indices_y]
    tfbs_mask = np.array(['tfbs' in label for label in sorted_labels])
    dnase_mask = np.array(['dnase' in label for label in sorted_labels])
    if cell_type == 'K562':
        hm_mask = np.array(['e123-h' in label for label in sorted_labels])
    else:
        hm_mask = np.array(['e116-h' in label for label in sorted_labels])
    
    # plt.scatter(degrees[neg_mask], diff[neg_mask],color='#ff0055',s=20)
    # plt.scatter(degrees[pos_mask], diff[pos_mask],color='#00c26e',s=20)

    tfbs_pos_mask = tfbs_mask*pos_mask
    tfbs_neg_mask = tfbs_mask*neg_mask
    dnase_pos_mask = dnase_mask*pos_mask
    dnase_neg_mask = dnase_mask*neg_mask
    hm_pos_mask = hm_mask*pos_mask
    hm_neg_mask = hm_mask*neg_mask

    plt.scatter(degrees[tfbs_pos_mask], diff[tfbs_pos_mask],color='#00c26e',s=20,label='TF')
    plt.scatter(degrees[tfbs_neg_mask], diff[tfbs_neg_mask],color='#ff0055',s=20)

    plt.scatter(degrees[hm_pos_mask], diff[hm_pos_mask],color='#00c26e',marker='^',s=20,label='HM')
    plt.scatter(degrees[hm_neg_mask], diff[hm_neg_mask],color='#ff0055',marker='^',s=20)
    
    plt.scatter(degrees[dnase_pos_mask], diff[dnase_pos_mask],color='#00c26e',marker='x',s=20,label='DNase I')
    plt.scatter(degrees[dnase_neg_mask], diff[dnase_neg_mask],color='#ff0055',marker='x',s=20)

    leg = plt.legend(loc='upper right',framealpha=0.4)
    LH = leg.legendHandles
    for handle in LH: handle.set_color('k')

    if metric == 'Recall_at_50':
        label_metric = 'Recall at 50% FDR'
    else:
        label_metric = metric

    ax.set_title(label_metric+' ('+cell_type+')',fontsize=15)
    plt.xlabel('Average Degree',fontsize=15)
    plt.ylabel('ChromeGCN$_{HiC}$ $-$ CNN',fontsize=15)
    
    # plt.ylabel('GNN (Hi-C) '+label_metric+' - CNN '+label_metric)
    # plt.setp(ax.get_yticklabels(),fontsize=3)
    plt.tight_layout()

    slope, intercept, r_value, p_value, std_err = stats.linregress(degrees,diff)

    z = np.polyfit(degrees, diff, 1)
    p = np.poly1d(z)
    plt.plot(degrees,p(degrees),"cornflowerblue",linewidth=2)

    # plt.annotate(r'r$^2$='+str(round(r_value**2,2)), xy=(degrees[-3],p(degrees)[-2]), xycoords='data',fontsize=8)
    if variation == 'GNN':
        ax.set_ylim(-0.05, 0.2)
    else:
        ax.set_ylim(-0.1, 0.2)
    # ax.set_aspect(1.0/ax.get_data_ratio()*1.0)
  
    plt.savefig(viz_path+cell_type+'_'+variation.replace(' ','_')+'_'+metric+'_diff.pdf',figsize=(512,512),dpi=1024,bbox_inches = 'tight', pad_inches = 0.15)
    # if metric == 'AUPR':
    #     stop()



def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value


def violin_plot(data,metric,viz_path,cell_type):
    fig, ax = plt.subplots()
    ax.set_title(cell_type)
    parts = ax.violinplot(data.T, showmeans=False, showmedians=False,showextrema=False)
    # parts['cmeans'].set_color('w')

    colors = mpl.cm.rainbow(np.linspace(0, 1, len(parts['bodies'])))
    
    for idx,pc in enumerate(parts['bodies']): pc.set_facecolor(colors[idx])
    for idx,pc in enumerate(parts['bodies']): pc.set_edgecolor('black')
    for idx,pc in enumerate(parts['bodies']): pc.set_alpha(1)

    quartile1, medians, quartile3 = np.percentile(data, [25, 50, 75], axis=1)
    means = np.mean(data, axis=1)
    whiskers = np.array([adjacent_values(sorted_array, q1, q3) for sorted_array, q1, q3 in zip(data, quartile1, quartile3)])
    whiskersMin, whiskersMax = whiskers[:, 0], whiskers[:, 1]

    inds = np.arange(1, len(medians) + 1)
    ax.scatter(inds, medians, marker='s', color='white', s=5, zorder=3)
    ax.scatter(inds, means, marker='o', color='white', s=5, zorder=3)
    ax.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
    ax.vlines(inds, whiskersMin, whiskersMax, color='k', linestyle='-', lw=1)

    plt.ylabel(metric.replace('_',' '))

    # ax.set_xticklabels(['','CNN','LSTM','GCN (no edges)','GCN (const.)','GCN (hi-c)','GCN (const.+hi-c)'])
    ax.set_xticklabels(['','CNN','LSTM','GCN (const.)','GCN (hi-c)','GCN (const.+hi-c)'])

    plt.setp(ax.get_xticklabels(),fontsize=8)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(viz_path+cell_type+'_'+metric+'_boxplot.pdf',figsize=(512,1024),dpi=1024)


def get_label_weights(opt,test_predictions,test_targets):
    adj = pickle.load( open( '/bigtemp/jjl5sw/ChromeGCN/data/'+args.cell_type+'/hic/'+'test'+'_graphs_min1000_samples'+args.hicsize+'_'+args.hicnorm+'norm.pkl', "rb" ) )
    
    # if not os.path.exists('/bigtemp/jjl5sw/ChromeGCN/data/'+args.cell_type+'/hic/test.pt'):
    # if True:
    #     data_dict = torch.load('/bigtemp/jjl5sw/ChromeGCN/data/'+args.cell_type+'/train_valid_test.pt')
    #     torch.save(data_dict['test'],'/bigtemp/jjl5sw/ChromeGCN/data/'+args.cell_type+'/test.pt')
    #     torch.save(data_dict['dict'],'/bigtemp/jjl5sw/ChromeGCN/data/'+args.cell_type+'/src_tgt_dict.pt')
    # else:
    #     test_data = torch.load('/bigtemp/jjl5sw/ChromeGCN/data/'+args.cell_type+'/test.pt')
    #     data_dict = torch.load('/bigtemp/jjl5sw/ChromeGCN/data/'+args.cell_type+'/src_tgt_dict.pt')
    test_data = torch.load('/bigtemp/jjl5sw/ChromeGCN/data/'+args.cell_type+'/test.pt')

    chrom_index_dict = {}
    for idx,sample in enumerate(test_data['loc']):
        chrom = sample[0]
        if not chrom in chrom_index_dict: 
            chrom_index_dict[chrom] = []
        chrom_index_dict[chrom].append(idx)

    test_labels = torch.Tensor(test_data['tgt'])
    label_neighbor_count = torch.zeros(len(test_data['tgt'][0]))
    label_count = torch.zeros(len(test_data['tgt'][0]))

    for chrom in chrom_index_dict:
        chrom_indices = torch.Tensor(chrom_index_dict[chrom]).long()
        chrom_labels = torch.index_select(test_labels, 0, chrom_indices)

        chrom_adj = utils.sparse_mx_to_torch_sparse_tensor(adj[chrom].tocoo())
        chrom_adj_d = chrom_adj.to_dense()
        chrom_adj_d[chrom_adj_d>1] = 1

        for idx,sample_labels in enumerate(chrom_labels):
            sample_labels_nz = sample_labels.nonzero()
            sample_neighbors = chrom_adj_d[idx].sum()
            label_neighbor_count[sample_labels_nz] += sample_neighbors
            label_count[sample_labels_nz] +=1

    normalized_label_weights =  label_neighbor_count.div(label_count)

    return normalized_label_weights

def analyze_individual_tfs(opt,test_predictions,test_targets):
    adj = pickle.load( open( '/bigtemp/jjl5sw/ChromeGCN/data/'+args.cell_type+'/hic/'+'test'+'_graphs_min1000_samples'+args.hicsize+'_'+args.hicnorm+'norm.pkl', "rb" ) )
    
    data_dict = torch.load('/bigtemp/jjl5sw/ChromeGCN/data/'+args.cell_type+'/train_valid_test.pt')
    torch.save(data['test'],'/bigtemp/jjl5sw/ChromeGCN/data/'+args.cell_type+'/test.pt')
    torch.save(data['dict'],'/bigtemp/jjl5sw/ChromeGCN/data/'+args.cell_type+'/src_tgt_dict.pt')

    
    chrom_index_dict = {}
    for idx,sample in enumerate(test_data['loc']):
        chrom = sample[0]
        if not chrom in chrom_index_dict: 
            chrom_index_dict[chrom] = []
        chrom_index_dict[chrom].append(idx)

    test_labels = torch.Tensor(test_data['tgt'])
    label_neighbor_count = torch.zeros(len(test_data['tgt'][0]))
    label_count = torch.zeros(len(test_data['tgt'][0]))
    chroms_auc_array,chroms_aupr_array,chroms_fdr_array = [],[],[]

    for chrom in chrom_index_dict:
        chrom_indices = torch.Tensor(chrom_index_dict[chrom]).long()
        chrom_labels = torch.index_select(test_labels, 0, chrom_indices)

        chrom_adj = utils.sparse_mx_to_torch_sparse_tensor(adj[chrom].tocoo())
        chrom_adj_d = chrom_adj.to_dense()
        chrom_adj_d[chrom_adj_d>1] = 1

        for idx,sample_labels in enumerate(chrom_labels):
            sample_labels_nz = sample_labels.nonzero()
            sample_neighbors = chrom_adj_d[idx].sum()
            label_neighbor_count[sample_labels_nz] += sample_neighbors
            label_count[sample_labels_nz] +=1

        chrom_test_predictions = torch.index_select(test_predictions, 0, chrom_indices)
        chrom_test_targets = torch.index_select(test_targets, 0, chrom_indices)
        chrom_test_metrics = compute_metrics(test_predictions,test_targets,0,opt,0,data_dict,args.cell_type)
        chroms_auc_array.append(chrom_test_metrics['meanAUC'])
        chroms_aupr_array.append(chrom_test_metrics['allAUPR'])
        chroms_fdr_array.append(chrom_test_metrics['allFDR'])

    normalized_label_weights =  label_neighbor_count.div(label_count)

    chroms_auc_array = np.array(chroms_auc_array)
    fig, ax = plt.subplots()
    chroms = [chrom for chrom in chrom_index_dict.keys()]
    ax.set_title(args.cell_type)
    x_pos = np.arange(len(chroms_auc_array)) 
    plt.bar(chroms, chroms_auc_array, align='center',color='b')
    plt.bar(chroms, chroms_auc_array, align='center',color='b')
    plt.xticks(x_pos, chroms)
    plt.ylabel('Mean AUC')
    plt.setp(ax.get_yticklabels(),fontsize=10)
    plt.tight_layout()
    # plt.savefig(args.viz_path+args.cell_type+'_'+variation.replace(' ','_')+'_'+metric+'_chrom_diffs_AUC.pdf',figsize=(512,512),dpi=1024)
    return normalized_label_weights

def main(opt):
    

    print('loading data')
    # if not os.path.exists('/bigtemp/jjl5sw/ChromeGCN/data/'+args.cell_type+'/hic/src_tgt_dict.pt'):
    # if True:
    #     data_dict = torch.load('/bigtemp/jjl5sw/ChromeGCN/data/'+args.cell_type+'/train_valid_test.pt')
    #     data_dict= data_dict['dict']
    #     torch.save(data_dict,'/bigtemp/jjl5sw/ChromeGCN/data/'+args.cell_type+'/src_tgt_dict.pt')
    # else:
    #     data_dict = torch.load('/bigtemp/jjl5sw/ChromeGCN/data/'+args.cell_type+'/src_tgt_dict.pt')
    print('loaded data')

    for variation,models in variations.items():
        print(variation)
        header = ''
        for model in models:
            header += model +','
        # header +='\n'
        # print('====================================================================================')
        # print('Variation: '+str(variation))
        csv_file = open(path.join(args.save_path,args.dataset+'_'+args.cell_type+'_'+str(variation)+'.csv'),'w+')

        csv_file.write(',TF,,,HM,,,DNase,,,All,,,\n')
        csv_file.write('model')
        # for metric in METRICS: csv_file.write(','+metric)
        csv_file.write(',mean auroc,mean aupr,mean fdr\
                        ,mean auroc,mean aupr,mean fdr\
                        ,mean auroc,mean aupr,mean fdr\
                        ,mean auroc,mean aupr,mean fdr\n')
        
        all_auc_array = []
        all_aupr_array = []
        all_fdr_array = []
        for model in models:
            print(model)

            csv_file.write(model)

            pred_files_dir = path.join(root,args.dataset,args.cell_type,model)
            valid_predictions = torch.load(path.join(pred_files_dir,'epochs','best_valid_preds_metrics.pt'))
            valid_targets = torch.load(path.join(pred_files_dir,'epochs','best_valid_targets_metrics.pt'))
            test_predictions = torch.load(path.join(pred_files_dir,'epochs','best_test_preds_metrics.pt'))
            test_targets = torch.load(path.join(pred_files_dir,'epochs','best_test_targets_metrics.pt'))
            # valid_predictions = torch.load(path.join(pred_files_dir,'epochs','best_valid_preds_loss.pt'))
            # valid_targets = torch.load(path.join(pred_files_dir,'epochs','best_valid_targets_loss.pt'))
            # test_predictions = torch.load(path.join(pred_files_dir,'epochs','best_test_preds_loss.pt'))
            # test_targets = torch.load(path.join(pred_files_dir,'epochs','best_test_targets_loss.pt'))
            losses = np.genfromtxt(path.join(pred_files_dir,'losses.csv'), delimiter=',')



            # if os.path.exists(path.join(root,args.dataset,args.cell_type,model,'test_metrics.pt')):
            if False:
                print('Loading Results Dict')
                # valid_metrics = torch.load(path.join(root,args.dataset,args.cell_type,model,'valid_metrics.pt'))
                test_metrics = torch.load(path.join(root,args.dataset,args.cell_type,model,'test_metrics.pt'))
            else:
                for threshold in THRESHOLDS: 
                    args.br_threshold = threshold

                    # valid_metrics = compute_metrics(valid_predictions,valid_targets,0,opt,0,data_dict,args.cell_type)
                    test_metrics = compute_metrics(test_predictions,test_targets,0,opt,0,data_dict,args.cell_type,per_label_type=True,plot=(not args.plot_diffs),model_name=model)

                    if (('finetune' in model) and (len(losses) > 999)) or (len(losses) > 99):
                        print('Saving Results Dict')
                        # torch.save(valid_metrics,path.join(root,args.dataset,args.cell_type,model,'valid_metrics.pt'))
                        torch.save(test_metrics,path.join(root,args.dataset,args.cell_type,model,'test_metrics.pt'))


            csv_file.write(','+str(round(test_metrics['tfbs_meanAUC'],3)))
            csv_file.write(','+str(round(test_metrics['tfbs_meanAUPR'],3)))
            csv_file.write(','+str(round(test_metrics['tfbs_meanFDR'],3)))

            csv_file.write(','+str(round(test_metrics['hm_meanAUC'],3)))
            csv_file.write(','+str(round(test_metrics['hm_meanAUPR'],3)))
            csv_file.write(','+str(round(test_metrics['hm_meanFDR'],3)))

            csv_file.write(','+str(round(test_metrics['dnase_meanAUC'],3)))
            csv_file.write(','+str(round(test_metrics['dnase_meanAUPR'],3)))
            csv_file.write(','+str(round(test_metrics['dnase_meanFDR'],3)))

            csv_file.write(','+str(round(test_metrics['meanAUC'],3)))
            csv_file.write(','+str(round(test_metrics['meanAUPR'],3)))
            csv_file.write(','+str(round(test_metrics['meanFDR'],3)))

            print('')
            
            all_auc_array.append(test_metrics['allAUC'])
            all_aupr_array.append(test_metrics['allAUPR'])
            all_fdr_array.append(test_metrics['allFDR'])

            csv_file.write('\n')


        all_auc_array = np.array(all_auc_array)
        all_aupr_array = np.array(all_aupr_array)
        all_fdr_array = np.array(all_fdr_array)

        
        if args.plot_diffs:
            label_weights= get_label_weights(opt,test_predictions,test_targets)
            plot_label_difference(all_auc_array[0], all_auc_array[1],'AUROC',variation,args.viz_path,args.cell_type,norm_weights=label_weights)
            plot_label_difference(all_aupr_array[0], all_aupr_array[1],'AUPR',variation,args.viz_path,args.cell_type,norm_weights=label_weights)
            plot_label_difference(all_fdr_array[0], all_fdr_array[1],'Recall_at_50',variation,args.viz_path,args.cell_type,norm_weights=label_weights)

        if args.plot_comparisons:
            plot_comparison(all_auc_array[0], all_auc_array[1],'AUROC',variation,args.viz_path,args.cell_type)
            plot_comparison(all_aupr_array[0], all_aupr_array[1],'AUPR',variation,args.viz_path,args.cell_type)
            plot_comparison(all_fdr_array[0], all_fdr_array[1],'Recall_at_50',variation,args.viz_path,args.cell_type)
        
        if args.violin_plots:
            violin_plot(all_auc_array,'AUC',args.viz_path,args.cell_type)
            violin_plot(all_aupr_array,'AUPR',args.viz_path,args.cell_type)
            violin_plot(all_fdr_array,'Recall_at_50',args.viz_path,args.cell_type) 

            cmd = 'convert '+args.viz_path+args.cell_type+'_AUC_boxplot.pdf ' \
                + args.viz_path+args.cell_type+'_AUPR_boxplot.pdf ' \
                + args.viz_path+args.cell_type+'_Recall_at_50_boxplot.pdf ' \
                + '+append '+ args.viz_path+args.cell_type+'_violin_plots.pdf'
            os.system(cmd)

        

        np.savetxt(args.save_path+'/auc.csv', all_auc_array.T, fmt='%.18e', delimiter=',', newline='\n', header=header,comments='')
        np.savetxt(args.save_path+'/aupr.csv', all_aupr_array.T, fmt='%.18e', delimiter=',', newline='\n', header=header,comments='')
        np.savetxt(args.save_path+'/fdr.csv', all_fdr_array.T, fmt='%.18e', delimiter=',', newline='\n', header=header,comments='')



  
if __name__== "__main__":
  main(opt)