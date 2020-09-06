import csv
import math
import os, glob
import sys
from random import randint, choice
import collections
import pickle
from pdb import set_trace as stop
from tqdm import tqdm
import numpy as np
from scipy import sparse
import pickle

def create_bin_dict(args,all_peaks_file_name):
	"""
	Input: global peaks file
	Output: global dictionary of bins with labels in the form of:
		bin_dict[chrom][start_pos]['pos_assays'][assay_id] = pvalue
	"""
	bin_dict = {} 
	for chrom in args.chroms:
		bin_dict[chrom]={}

	with open(all_peaks_file_name) as csvfile:
		csv_reader = csv.DictReader(csvfile,delimiter='\t',fieldnames=['chrom','start_pos','end_pos','assay_id','score','strand','signalValue','pvalue','qValue','peak'])
		for csv_row in csv_reader:
			chrom = str(csv_row['chrom'])
			start_pos = int(csv_row['start_pos'])
			assay_id = str(csv_row['assay_id'])
		
			if chrom in args.chroms:
				if start_pos not in bin_dict[chrom]:
					bin_dict[chrom][start_pos] = {}
					# bin_dict[chrom][start_pos]['bin_idx'] = len(bin_dict[chrom])-1
					# bin_dict[chrom][start_pos]['pos_assays'] = {}

				# bin_dict[chrom][start_pos]['pos_assays'][assay_id] = 1

	# sort each chromosome by start position
	for chrom in bin_dict: 
		bin_dict[chrom] = collections.OrderedDict(sorted(bin_dict[chrom].items()))

		for idx,start_pos in enumerate(bin_dict[chrom]):
			bin_dict[chrom][start_pos]['bin_idx'] = idx
	

	return bin_dict



def get_normalization_values(hic_norm_file,chrom):
	"""
	Get normalization values for each distance between windows
	"""
	normalization_values = []
	with open(hic_norm_file) as csvfile:
		csv_reader = csv.DictReader(csvfile,delimiter='\t',fieldnames=['norm_value'])
		for csv_row in csv_reader:
			normalization_values.append(float(csv_row['norm_value']))
	# normalization_values = [1 if math.isnan(x) else x for x in normalization_values]
	# normalization_values = [1 if x==0.0 else x for x in normalization_values]
	normalization_values = [float("inf") if math.isnan(x) else x for x in normalization_values]
	normalization_values = [float("inf") if x==0.0 else x for x in normalization_values]
	
	return normalization_values

def get_contact_edge_pairs(args,hic_file,chrom,normalization_values,residuals,bin_dict,total_edges):
	contact_edge_pairs = {}
	total_count = 0
	
	with open(hic_file) as csvfile:
		csv_reader = csv.DictReader(csvfile,delimiter='\t',fieldnames=['start_pos1','start_pos2','val'])
		for csv_row in tqdm(csv_reader,total=126204768):
			bin1=int(csv_row['start_pos1'])
			bin2=int(csv_row['start_pos2'])
			val = float(csv_row['val'])

			if (bin1 != bin2) and (bin1 in bin_dict[chrom]) and (bin2 in bin_dict[chrom]):
				total_count +=1
				if normalization_values is not None:
					norm_denom1 = normalization_values[int(int(bin1)/(1000*int(args.resolution)))]
					norm_denom2 = normalization_values[int(int(bin2)/(1000*int(args.resolution)))]

					val = val/(norm_denom1*norm_denom2)
				
				contact_edge_pairs[(bin1,bin2)] = val
				
				if args.norm == '' and (total_count == total_edges):
					return contact_edge_pairs

	return contact_edge_pairs

def get_top_contact_locs(contact_edge_pairs,total_edges):
	sorted_contact_edge_pairs =  {k: v for k, v in sorted(contact_edge_pairs.items(), key=lambda item: item[1], reverse=True)}

	top_contact_edge_pairs = {}
	idx = 0
	for (bin1,bin2),val in sorted_contact_edge_pairs.items():
		idx +=1
		top_contact_edge_pairs[(bin1,bin2)] = val
		if idx == total_edges:
			break
		
	return top_contact_edge_pairs



def create_adj_mat(bin_dict,chrom,top_contact_locs):
	adj_mat = np.zeros((len(bin_dict[chrom]),len(bin_dict[chrom])))

	for bin1,bin2 in top_contact_locs:
		bin1idx = bin_dict[chrom][bin1]['bin_idx']
		bin2idx = bin_dict[chrom][bin2]['bin_idx']

		adj_mat[bin1idx,bin2idx] = 1
		adj_mat[bin2idx,bin1idx] = 1

	sparse_adj_mat = sparse.csr_matrix(adj_mat)

	return sparse_adj_mat


def print_bin_sizes(args,bin_dict):
	train_total,valid_total,test_total = 0,0,0
	for chrom in args.chroms: 
		if chrom in args.test_chroms:
			test_total += len(bin_dict[chrom])
		elif chrom in args.valid_chroms:
			valid_total += len(bin_dict[chrom])
		else:
			train_total += len(bin_dict[chrom])
	print('Train Total Size:',train_total)
	print('Valid Total Size:',valid_total)
	print('Test Total Size:',test_total)

def create_graph(args):

	output_root = args.output_root

	if args.use_all_windows:
		all_peaks_file_name = os.path.join(args.output_root,'windows.bed')
	else:
		all_peaks_file_name = os.path.join(args.output_root,'chipseq_windows.bed')

	hic_root = os.path.join(args.hic_root,args.cell_type+'_combined',args.resolution+'kb_resolution_intrachromosomal/')
	
	train_dict = os.path.join(output_root,'hic/train_graphs'+'_'+str(args.hic_edges)+'_'+args.norm+'norm.pkl')
	valid_dict = os.path.join(output_root,'hic/valid_graphs'+'_'+str(args.hic_edges)+'_'+args.norm+'norm.pkl')
	test_dict = os.path.join(output_root,'hic/test_graphs'+'_'+str(args.hic_edges)+'_'+args.norm+'norm.pkl')

	print('\nInputs')
	print('| '+all_peaks_file_name)
	print('| '+hic_root)
	print('\nOutputs')
	print('| '+train_dict)
	print('| '+valid_dict)
	print('| '+test_dict)

	train_idx_dict = {}
	valid_idx_dict = {}
	test_idx_dict = {}
	
	# print('--------Part0-----------')
	print('\nCreate Global Bin Dictionary')
	bin_dict = create_bin_dict(args,all_peaks_file_name)
	print_bin_sizes(args,bin_dict)

	total_edges = int(args.hic_edges/2.)
	for chrom in args.chroms:
		print(chrom)
		
		# print('--------Part1-----------')
		if args.norm != '':
			hic_norm_file = os.path.join(hic_root,chrom+'/MAPQGE30/'+chrom+'_'+args.resolution+'kb.'+args.norm+'norm')
			hic_file = os.path.join(hic_root,chrom+'/MAPQGE30/'+chrom+'_'+args.resolution+'kb.RAWobserved')
			normalization_values = get_normalization_values(hic_norm_file,chrom)
		else:
			hic_file = os.path.join(hic_root,chrom+'/MAPQGE30/'+chrom+'_'+args.resolution+'kb.RAWobserved.sorted')
			normalization_values = None

		# print('--------Part2-----------')
		contact_edge_pairs = get_contact_edge_pairs(args,hic_file,chrom,normalization_values,args.residuals,bin_dict,total_edges)

		# # print('--------Part3-----------')
		top_contact_edge_pairs = get_top_contact_locs(contact_edge_pairs,total_edges)

		# print('--------Part4-----------')
		sparse_adj_mat = create_adj_mat(bin_dict,chrom,top_contact_edge_pairs)

		if chrom in args.test_chroms:
			test_idx_dict[chrom] = sparse_adj_mat
		elif chrom in args.valid_chroms:
			valid_idx_dict[chrom] = sparse_adj_mat
		else:
			train_idx_dict[chrom] = sparse_adj_mat

	with open(train_dict, "wb") as fp: 
		pickle.dump(train_idx_dict, fp)
	with open(valid_dict, "wb") as fp: 
		pickle.dump(valid_idx_dict, fp)
	with open(test_dict, "wb") as fp: 
		pickle.dump(test_idx_dict, fp)



