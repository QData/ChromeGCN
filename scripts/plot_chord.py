
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import numpy as np
import pickle
import os
from scipy import sparse  
from pdb import set_trace as stop

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-chrom', type=str, default='chr8')
parser.add_argument('-cell_type', type=str, default='GM12878')
parser.add_argument('-norm', type=str, choices=['KR','VC','SQRTVC',''], default='SQRTVC')
parser.add_argument('-size', type=int, default=500000)
parser.add_argument('-save_path', type=str, default='./chord_diagrams')
parser.add_argument('-gate_weight', action='store_true')
parser.add_argument('-A_saliency', action='store_true')



opt = parser.parse_args()

print(opt.chrom)

if not os.path.exists(opt.save_path):
    os.makedirs(opt.save_path)

valid_chroms = ['chr3', 'chr12', 'chr17']
test_chroms = ['chr1', 'chr8', 'chr21']

LW = 0.01

def polar2xy(r, theta):
    return np.array([r*np.cos(theta), r*np.sin(theta)])

def hex2rgb(c):
    return tuple(int(c[i:i+2], 16)/256.0 for i in (1, 3 ,5))

def IdeogramArc(start=0, end=60, radius=1.0, width=0.2, ax=None, color=(1,0,0)):
    # start, end should be in [0, 360)
    if start > end:
        start, end = end, start
    start *= np.pi/180.
    end *= np.pi/180.
    # optimal distance to the control points
    # https://stackoverflow.com/questions/1734745/how-to-create-circle-with-b%C3%A9zier-curves
    opt = 4./3. * np.tan((end-start)/ 4.) * radius
    inner = radius*(1-width)
    verts = [
        polar2xy(radius, start),
        polar2xy(radius, start) + polar2xy(opt, start+0.5*np.pi),
        polar2xy(radius, end) + polar2xy(opt, end-0.5*np.pi),
        polar2xy(radius, end),
        polar2xy(inner, end),
        polar2xy(inner, end) + polar2xy(opt*(1-width), end-0.5*np.pi),
        polar2xy(inner, start) + polar2xy(opt*(1-width), start+0.5*np.pi),
        polar2xy(inner, start),
        polar2xy(radius, start),
        ]

    codes = [Path.MOVETO,
             Path.CURVE4,
             Path.CURVE4,
             Path.CURVE4,
             Path.LINETO,
             Path.CURVE4,
             Path.CURVE4,
             Path.CURVE4,
             Path.CLOSEPOLY,
             ]

    if ax == None:
        return verts, codes
    else:
        path = Path(verts, codes)
        patch = patches.PathPatch(path, facecolor=color+(0.5,), edgecolor=color+(0.5,), lw=LW)
        ax.add_patch(patch)


def ChordArc(start1=0, end1=60, start2=180, end2=240, gate_val=1, radius=1.0, chordwidth=0.7, ax=None, color=(1,0,0)):
    # start, end should be in [0, 360)
    if start1 > end1:
        start1, end1 = end1, start1
    if start2 > end2:
        start2, end2 = end2, start2
    start1 *= np.pi/180.
    end1 *= np.pi/180.
    start2 *= np.pi/180.
    end2 *= np.pi/180.
    opt1 = 4./3. * np.tan((end1-start1)/ 4.) * radius
    opt2 = 4./3. * np.tan((end2-start2)/ 4.) * radius
    rchord = radius * (1-chordwidth)
    verts = [
        polar2xy(radius, start1),
        polar2xy(radius, start1) + polar2xy(opt1, start1+0.5*np.pi),
        polar2xy(radius, end1) + polar2xy(opt1, end1-0.5*np.pi),
        polar2xy(radius, end1),
        polar2xy(rchord, end1),
        polar2xy(rchord, start2),
        polar2xy(radius, start2),
        polar2xy(radius, start2) + polar2xy(opt2, start2+0.5*np.pi),
        polar2xy(radius, end2) + polar2xy(opt2, end2-0.5*np.pi),
        polar2xy(radius, end2),
        polar2xy(rchord, end2),
        polar2xy(rchord, start1),
        polar2xy(radius, start1),
        ]

    codes = [Path.MOVETO,
             Path.CURVE4,
             Path.CURVE4,
             Path.CURVE4,
             Path.CURVE4,
             Path.CURVE4,
             Path.CURVE4,
             Path.CURVE4,
             Path.CURVE4,
             Path.CURVE4,
             Path.CURVE4,
             Path.CURVE4,
             Path.CURVE4,
             ]

    if ax == None:
        return verts, codes
    else:
        path = Path(verts, codes)
        patch = patches.PathPatch(path, facecolor=color+(gate_val,), edgecolor=color+(gate_val,), lw=LW)
        ax.add_patch(patch)

def selfChordArc(start=0, end=60, radius=1.0, chordwidth=0.7, ax=None, color=(1,0,0)):
    # start, end should be in [0, 360)
    if start > end:
        start, end = end, start
    start *= np.pi/180.
    end *= np.pi/180.
    opt = 4./3. * np.tan((end-start)/ 4.) * radius
    rchord = radius * (1-chordwidth)
    verts = [
        polar2xy(radius, start),
        polar2xy(radius, start) + polar2xy(opt, start+0.5*np.pi),
        polar2xy(radius, end) + polar2xy(opt, end-0.5*np.pi),
        polar2xy(radius, end),
        polar2xy(rchord, end),
        polar2xy(rchord, start),
        polar2xy(radius, start),
        ]

    codes = [Path.MOVETO,
             Path.CURVE4,
             Path.CURVE4,
             Path.CURVE4,
             Path.CURVE4,
             Path.CURVE4,
             Path.CURVE4,
             ]

    if ax == None:
        return verts, codes
    else:
        path = Path(verts, codes)
        patch = patches.PathPatch(path, facecolor=color, edgecolor=color, lw=LW)
        ax.add_patch(patch)

def chordDiagram(X, ax, gate_vals=None,colors=None, width=0.1, pad=2, chordwidth=0.7):
    """Plot a chord diagram

    Parameters
    ----------
    X :
        flux data, X[i, j] is the flux from i to j
    ax :
        matplotlib `axes` to show the plot
    colors : optional
        user defined colors in rgb format. Use function hex2rgb() to convert hex color to rgb color. Default: d3.js category10
    width : optional
        width/thickness of the ideogram arc
    pad : optional
        gap pad between two neighboring ideogram arcs, unit: degree, default: 2 degree
    chordwidth : optional
        position of the control points for the chords, controlling the shape of the chords
    """
    # X[i, j]:  i -> j
    x = X.sum(axis = 1) # sum over rows
    x[x==0] = 1
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)

    if colors is None:
        # colors=['#ff0040'] # Red
        colors=['#4700b3'] # Purple
    
        
        colors = [hex2rgb(colors[0])]

    # find position for each start and end
    y = x/np.sum(x).astype(float) * (360 - pad*len(x))

    pos = {}
    arc = []
    nodePos = []
    start = 0
    
    for i in range(len(x)):
    # nz = X.nonzero()
    # for i in range(len(x)):
        # print(str(i)+'/'+str(len(x)))
        end = start + y[i]
        arc.append((start, end))
        angle = 0.5*(start+end)
        #print(start, end, angle)
        if -30 <= angle <= 210:
            angle -= 90
        else:
            angle -= 270
        nodePos.append(tuple(polar2xy(1.1, 0.5*(start+end)*np.pi/180.)) + (angle,))
        z = (X[i, :]/x[i].astype(float)) * (end - start)
        z0 = start
        for j in z.nonzero()[0]:
        # ids = np.argsort(z)
        # for j in ids:
            pos[(i, j)] = (z0, z0+z[j])
            z0 += z[j]

        start = end + pad

    # stop()
    for i in range(len(x)):
        # print(str(i)+'/'+str(len(x)))
        start, end = arc[i]
        IdeogramArc(start=start, end=end, radius=1.0, ax=ax, color=colors[0], width=width)
        # start, end = pos[(i,i)]
        # selfChordArc(start, end, radius=1.-width, color=colors[0], chordwidth=chordwidth, ax=ax)
        # stop()
        for j in X[i].nonzero()[0]:
        # for j in range(i):
            # color = colors[0]
            # if X[i, j] > X[j, i]:
            #     color = colors[j]
            if (i,j) in pos:
                start1, end1 = pos[(i,j)]
                start2, end2 = pos[(j,i)]

                if (start1 != end1) or (start2 != end2):
                    ChordArc(start1, end1, start2, end2,gate_val=gate_vals[i],radius=1.-width, color=colors[0], chordwidth=chordwidth, ax=ax)

    return nodePos

def chordDiagram_saliency(X, ax, weights=None,colors=None, width=0.1, pad=2, chordwidth=0.7):
    x = X.sum(axis = 1) # sum over rows
    x[x==0] = 1
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    if colors is None:
        # colors=['#ff0040']
        colors=['#4700b3'] # Purple
        colors = [hex2rgb(colors[0])]
    # find position for each start and end
    y = x/np.sum(x).astype(float) * (360 - pad*len(x))
    pos = {}
    arc = []
    nodePos = []
    start = 0
    for i in range(len(x)):
        end = start + y[i]
        arc.append((start, end))
        angle = 0.5*(start+end)
        if -30 <= angle <= 210:
            angle -= 90
        else:
            angle -= 270
        nodePos.append(tuple(polar2xy(1.1, 0.5*(start+end)*np.pi/180.)) + (angle,))
        z = (X[i, :]/x[i].astype(float)) * (end - start)
        z0 = start
        for j in z.nonzero()[0]:
            pos[(i, j)] = (z0, z0+z[j])
            z0 += z[j]
        start = end + pad
    for i in range(len(x)):
        start, end = arc[i]
        IdeogramArc(start=start, end=end, radius=1.0, ax=ax, color=colors[0], width=width)
        for j in X[i].nonzero()[0]:
            if (i,j) in pos:
                start1, end1 = pos[(i,j)]
                start2, end2 = pos[(j,i)]

                if (start1 != end1) or (start2 != end2):
                    ChordArc(start1, end1, start2, end2,gate_val=weights[i,j],radius=1.-width, color=colors[0], chordwidth=chordwidth, ax=ax)
    return nodePos


fig_size = 128
dpi=100

if opt.A_saliency:
    import torch
    print('saliency')
    chrom=opt.chrom
    adj = torch.load('chord_diagrams/GM12878_'+chrom+'_saliency.pt')
    # adj = torch.load('chord_diagrams/GM12878_'+chrom+'_saliency_yy1.pt')
    # print('chord_diagrams/GM12878_'+chrom+'_saliency_yy1.pt')
    adj = adj.detach()
    adj_nonzero = adj.clone()
    adj_nonzero[adj_nonzero>0] = 1
    adj = adj.numpy()
    adj_nonzero = adj_nonzero.numpy()
    # stop()

    fig = plt.figure(figsize=(fig_size,fig_size),dpi=100)
    ax = plt.axes([0,0,1,1])
    stop()
    nodePos = chordDiagram_saliency(adj_nonzero, ax, weights=adj,width=0.00001,chordwidth=0.00001,pad=0.0001)
    ax.axis('off')
    print('saving')
    plt.savefig(opt.save_path+'/'+chrom+"_"+opt.norm+"A_saliency_yy1.png",dpi=dpi,transparent=False)


    



elif opt.gate_weight:
    import torch
    gate_dict = torch.load('chord_diagrams/gate_saved.pt')
    adj = gate_dict['adj']
    gate_vals = gate_dict['gate'].numpy()
    chrom = gate_dict['chrom']
    adj = adj.tocoo()
    adj = adj.todense()
    adj = np.array(adj)
    adj = gate_vals.reshape(gate_vals.shape[0],1)*adj

    

    fig = plt.figure(figsize=(fig_size,fig_size),dpi=100,facecolor='black')
    ax = plt.axes([0,0,1,1])
    nodePos = chordDiagram(adj, ax, gate_vals=gate_vals,width=0.00001,chordwidth=0.00001,pad=0.0001)
    ax.axis('off')
    
    # plt.savefig(opt.save_path+'/'+chrom+"_"+opt.norm+"_gateweight.png",dpi=100,facecolor=fig.get_facecolor(),transparent=True)
    plt.savefig(opt.save_path+'/'+chrom+"_"+opt.norm+"_gateweight.png",dpi=100,transparent=False)

else:

    chroms = opt.chrom.split(',')
    for chrom in chroms:
        
        if chrom in valid_chroms:
            split = 'valid'
        elif chrom in test_chroms:
            split = 'test'
        else:
            split = 'train'

        
        test_adj = pickle.load( open( '/af11/jjl5sw/BERT/genomeGCN/data/encode/'+opt.cell_type+'/'+split+'_graphs_min1000_samples'+str(opt.size)+'_'+opt.norm+'norm.pkl', "rb" ) ) 
        test_adj = test_adj[chrom].tocoo() 
        test_adj = test_adj + test_adj.T.multiply(test_adj.T > test_adj) - test_adj.multiply(test_adj.T > test_adj)
        # test_adj = test_adj + sparse.eye(test_adj.shape[0]) 
        test_adj[test_adj>1] = 1
        test_adj = test_adj.tocoo()
        test_adj = test_adj.todense()
        test_adj = np.array(test_adj)
        gate_vals = np.ones(test_adj.shape[0])
        adj=test_adj

        
        # fig_size = 100
        # p0 = 0.99
        # p1 = 1-p0
        # size = 100
        # adj = np.random.choice([0, 1], size=((size,size)), p=[p0,p1])
        # adj[adj>1] = 1
        # for i in range(size):
        #     for j in range(size):
        #         if adj[i,j] == 1:
        #             adj[j,i] = 1
        #         if adj[j,i] == 1:
        #             adj[i,j] = 1
        # gate_vals = np.random.rand(size)
        # print(adj)

        print(adj.sum()) 
        # stop()
        fig = plt.figure(figsize=(fig_size,fig_size))
        ax = plt.axes([0,0,1,1])
        ax.set_facecolor((0,0,0))
        nodePos = chordDiagram(adj, ax, gate_vals=gate_vals,width=0.00001,chordwidth=0.00001,pad=0.00001)
        ax.axis('off')
        plt.savefig(opt.save_path+'/'+chrom+"_"+opt.norm+".png",transparent=False)



