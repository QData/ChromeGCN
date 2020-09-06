
!import code; code.interact(local=vars())
import argparse,math,time,warnings,copy,tables, pickle, numpy as np, os.path as path 
import evals,utils
import torch, torch.nn as nn, torch.nn.functional as F
from pdb import set_trace as stop
from tqdm import tqdm
from scipy import sparse
from utils import pad_batch,unpad_batch
import pandas as pd
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap



"""
######################################### A SALIENCY #####################################################
"""
# black_green = LinearSegmentedColormap.from_list('', colors=((0.0, 0.0, 0.0), (0.0, 1.0, 0)), N=500)
# cm = LinearSegmentedColormap.from_list('', colors=((1.0, 1.0, 1.0), (1.0, 0.0, 0.25)), N=500)
# cm = LinearSegmentedColormap.from_list('', colors=((0.0, 0.0, 0.0), (0.0, 1.0, 0),(1.0, 1.0, 0.0)), N=500)

# adj[adj>0] = 1
# adj=adj-torch.eye(adj.size(0))


gcn_model=gcn_model.cpu()
adj = split_adj.to_dense().cpu()
adj.requires_grad=True

gcn_model.zero_grad()
if adj.grad is not None: 
    adj.grad = adj.grad.zero_()

_,pred_f,_,_ = gcn_model(x_f.cpu(),adj.cpu(),None,src_dict=data_dict['src'])
_,pred_r,_,_ = gcn_model(x_r.cpu(),adj.cpu(),None,src_dict=data_dict['src'])
pred = (pred_f+pred_r)/2
sig_pred = F.sigmoid(pred)

# yy1_mask = targets.clone().zero_()
# yy1_mask[:,21]=1.0
# targets_yy1 = targets*yy1_mask
# sig_pred.backward(gradient=targets_yy1.cpu())

sig_pred.backward(gradient=targets.cpu())

adj_grad = torch.abs(adj*adj.grad)
sum_val = adj_grad.sum(1)
sum_val[sum_val==0] = 1
adj_grad=adj_grad/sum_val.view(-1,1)
max_val,_=torch.max(adj_grad,1)
max_val[max_val==0] = 1
adj_grad=adj_grad/max_val.view(-1,1)

# torch.save(adj_grad,'chord_diagrams/'+opt.cell_type+'_'+chrom+'_'+'saliency.pt')

dpi=1000
plt.figure(dpi=dpi)
ax = plt.axes()
plt.ylabel('Windows')
plt.xlabel('Windows')
start=10250
end=10500
ax.set_xticklabels([0]+list(range(start,end,int((end-start)/len(ax.get_xticklabels())))))
ax.set_yticklabels([0]+list(range(start,end,int((end-start)/len(ax.get_xticklabels())))))
plt.tight_layout()
plt.imshow(adj_grad.detach().numpy()[start:end,start:end], cmap='Purples',vmin=0,vmax=1)
plt.savefig('A_saliency2.pdf',dpi=dpi,bbox_inches='tight')


"""
###################################### TF-TF Relationships ############################################
"""

label_list = [key.replace('wgencodeawg','').replace('unipk','').replace('gm12878','').replace('k562','').replace('iggmus','').replace('syd','').replace('uta','').replace('haib','').replace('pcr1x','').replace('pcr2x','').replace('tfbs','tfbs_').replace('iggrab','').replace('broad','').replace('hchd2','').replace('143166181ap','').replace('v0416101','').replace('anb6001263','').replace('iknucla','').replace('ab9263','').replace('ab85725','').replace('ab24550','').replace('101388','').replace('239875','').replace('112771','').replace('a301218a','').split('sc')[0] for key in data_dict['tgt'].keys()]

tf_tf_mats=[]
targets=targets.cpu()
y_i=0
y_j=1
adj = split_adj.to_dense().cpu()
tf_tf_mat = torch.zeros(targets.size(-1),targets.size(-1))
zero_mat = torch.zeros(adj.size()).byte()
for y_i in range(10,80):
    y_i_pos_samples = targets[:,y_i].nonzero().view(-1)
    y_i_pos_sample_edges = adj[y_i_pos_samples,:]
    pred_i_mean = pred[y_i_pos_samples,y_i].sigmoid().mean().item()
    y_i_pos_mat = zero_mat.index_fill(0, y_i_pos_samples, 1)
    # neg_targets_i = targets[y_i_pos_samples]
    # neg_targets_i[neg_targets_i==0]=-1
    # neg_targets_i[neg_targets_i==1]=0
    # neg_targets_i[neg_targets_i==-1]=1
    # pred_i_neg_mean = (pred[y_i_pos_samples,:].sigmoid().cpu()*neg_targets_i).mean().item()
    # pred_i_mean = pred_i_mean-pred_i_neg_mean
    for y_j in range(10,80):
        print(str(y_i)+','+str(y_j))
        y_j_pos_samples = targets[:,y_j].nonzero().view(-1)
        if len(y_j_pos_samples)>0 and (y_i != y_j):
            y_j_pos_mat = zero_mat.index_fill(1, y_j_pos_samples, 1)
            mask = (y_i_pos_mat*y_j_pos_mat)
            adj2 = adj.masked_fill(mask, 0)
            sum_vals = adj2.sum(1).view(-1,1)
            sum_vals[sum_vals == 0] = 1
            adj2 = adj2/sum_vals
            indices = torch.nonzero(adj2).t()
            values = adj2[indices[0], indices[1]]
            adj2 = torch.sparse.FloatTensor(indices, values, adj2.size())
            _,pred_f_ij,_,_ = gcn_model(x_f,adj2.cuda(),None,src_dict=data_dict['src'])
            _,pred_r_ij,_,_ = gcn_model(x_r,adj2.cuda(),None,src_dict=data_dict['src'])
            pred_ij = (pred_f_ij+pred_r_ij)/2
            pred_ij_mean = pred_ij[y_i_pos_samples,y_i].sigmoid().mean().item()
            # pred_ij_neg_mean = (pred_ij[y_i_pos_samples,:].sigmoid().cpu()*neg_targets_i).mean().item()
            # pred_ij_mean = pred_ij_mean-pred_ij_neg_mean
            diff = (pred_i_mean-pred_ij_mean)/pred_i_mean
            tf_tf_mat[y_i,y_j] = diff
            print(diff)
tf_tf_mats.append(tf_tf_mat)


import matplotlib.pyplot as plt
tfbs_indices = [i for i,label in enumerate(label_list) if 'tfbs' in label]
tfbs_labels = [label.replace('tfbs_','') for i,label in enumerate(label_list) if 'tfbs' in label]
tf_tf_mat_plot = tf_tf_mat.index_select(0,torch.Tensor(tfbs_indices).long())
tf_tf_mat_plot = tf_tf_mat_plot.index_select(1,torch.Tensor(tfbs_indices).long())
tf_tf_mat_plot = tf_tf_mat_plot[0:50,0:50]
tfbs_labels = tfbs_labels[0:50]
tf_tf_mat_plot=tf_tf_mat_plot*100
# min_val,_=torch.min(tf_tf_mat_plot,1)
# max_val,_=torch.max(tf_tf_mat_plot,1)
# tf_tf_mat_plot = (tf_tf_mat_plot-(min_val.view(-1,1)))/(max_val.view(-1,1)-min_val.view(-1,1))
# tf_tf_mat_plot = (tf_tf_mat_plot*(1--1))-1
plt.figure()
plt.imshow(tf_tf_mat_plot, cmap='bwr',vmin=-tf_tf_mat_plot.max(),vmax=tf_tf_mat_plot.max())
plt.tick_params(axis=u'both', which=u'both',length=0)
plt.xticks(np.arange(tf_tf_mat_plot.size(0)), tfbs_labels,rotation=90,size=3)
plt.yticks(np.arange(tf_tf_mat_plot.size(1)), tfbs_labels,size=3)
plt.colorbar()
ax = plt.axes()
b, t = plt.ylim()
b += 0.5 
t -= 0.5 
plt.ylim(b, t) 
plt.savefig('tf_tf.pdf',dpi=500)


"""
###################################### TSNE ############################################
"""

all_single_class_z2 = (all_single_class_z[:,0:128]+all_single_class_z[:,128:])/2

label_list = [key.replace('wgencodeawg','').replace('unipk','').replace('gm12878','').replace('k562','').replace('iggmus','').replace('syd','').replace('uta','').replace('haib','').replace('pcr1x','').replace('pcr2x','').replace('tfbs','tfbs_').replace('iggrab','').replace('broad','').replace('hchd2','').replace('143166181ap','').replace('v0416101','').replace('anb6001263','').replace('iknucla','').replace('ab9263','').replace('ab85725','').replace('ab24550','').split('sc')[0] for key in data_dict['tgt'].keys()]
tfbs_indices = [i for i,label in enumerate(label_list) if 'tfbs' in label]
tfbs_labels = [label for i,label in enumerate(label_list) if 'tfbs' in label]
used_tfbs_labels = []


tsne_labels = []
tsne_inputs = torch.Tensor()
for i in tfbs_indices:
    i_nonzero = (all_single_class_targs==i).nonzero().view(-1)#[0:100]
    if len(i_nonzero)>199:
        print(len(i_nonzero))
        i_targets = all_single_class_targs[i_nonzero]
        i_z = all_single_class_z[i_nonzero].data
        tsne_labels += [tfbs_labels[i].replace('tfbs_','') for i in i_targets]
        used_tfbs_labels.append(tfbs_labels[i].replace('tfbs_',''))
        tsne_inputs = torch.cat((tsne_inputs,i_z),0)

# tsne_labels = tsne_labels.detach().numpy()
tsne_inputs = tsne_inputs.detach().numpy()

for perp in [5,10,15,20,25,30,35,40,45,50,55,60,65]:
    X_embedded = TSNE(n_components=2,perplexity=perp,n_iter=3000, n_iter_without_progress=500).fit_transform(tsne_inputs)
    labels = sorted(set(tsne_labels))
    colors = cm.rainbow(np.linspace(0,1,len(labels)))
    all_colors = [colors[labels.index(label)] for label in tsne_labels]
    label_indices = [labels.index(label) for label in tsne_labels]
    label_indices = np.array(label_indices)
    fig, ax = plt.subplots()
    scatter = [plt.scatter(X_embedded[label_indices == i, 0][0:-1], X_embedded[label_indices == i, 1][0:-1], s=1.5,c=c, label=label) for i, c, label in zip(range(len(X_embedded)), colors, labels)]
    plt.legend(bbox_to_anchor=(1.0,1.01), loc="upper left",prop={'size': 5})
    plt.subplots_adjust(right=0.7)
    fig.canvas.draw()
    plt.tight_layout()  
    fig.savefig('plot'+str(perp)+'_mean.png',figsize=(500,512),dpi=512)

"""
######################################################################################
"""