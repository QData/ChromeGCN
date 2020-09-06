import csv
import math
import os, glob
import sys
from random import randint, choice
import collections
import pickle
# import constants as Constants
from pdb import set_trace as stop
from tqdm import tqdm
import numpy as np
from scipy import sparse
import pickle

def create_graph(args):

	hg_file = args.hg_file

	chroms=args.chroms
	extended_window_length = args.extended_window_length
	stride_length=args.stride_length
	tad_file = args.tad_file

	chroms=args.chroms
	# chroms = ['chr3']

	output_root=args.output_root


	if args.cell_type == '':
		cell_type = args.cell_type
	else:
		cell_type = args.cell_type

	if cell_type == 'GM12878':
		args.resolution='1'
	else:
		args.resolution = '5'
		# args.min_distance_threshold=5000


	valid_chroms = args.valid_chroms
	test_chroms = args.test_chroms


	all_peaks_file_name = os.path.join(output_root,'chipseq_windows.bed')
	print('Input1: '+all_peaks_file_name)
	print('Input2: '+tad_file)
	print('Output: '+cell_type+'/train_graphs_min'+str(args.min_distance_threshold)+'_samples'+str(args.size)+'_'+args.norm+'norm.pkl')

	tads = {}
	# with open(tad_file) as csvfile:
	# 	csv_reader = csv.DictReader(csvfile,delimiter='\t',fieldnames=['chrom','start_pos','end_pos'])
	# 	for csv_row in csv_reader:
	# 		chrom = str(csv_row['chrom'])
	# 		start_pos = int(csv_row['start_pos'])
	# 		end_pos = int(csv_row['end_pos'])
			
	# 		if not chrom in tads:
	# 			tads[chrom] = []

	# 		tads[chrom].append((start_pos,end_pos))


	global_dict = {} #Format: global_dict[chrom][start_pos]['TFs'][TF_id] = pvalue
	global_dict_count = {}
	global_dict_1000 = {}
	for chrom in chroms:
		global_dict[chrom]={}
		global_dict_count[chrom]=0
		global_dict_1000[chrom]={}


	TF_ids = []

	with open(all_peaks_file_name) as csvfile:
		csv_reader = csv.DictReader(csvfile,delimiter='\t',fieldnames=['chrom','start_pos','end_pos','TF_id','score','strand','signalValue','pvalue','qValue','peak'])
		for csv_row in csv_reader:
			TF_id = str(csv_row['TF_id'])
			chrom = str(csv_row['chrom'])
			start_pos = int(csv_row['start_pos'])

			if chrom in chroms:

				if not TF_id in TF_ids:
					TF_ids.append(TF_id)

				
				hic_bin = start_pos - (start_pos % 1000)
				if hic_bin not in global_dict_1000[chrom]:
					global_dict_1000[chrom][hic_bin] = None


				if not start_pos in global_dict[chrom]:
					global_dict[chrom][start_pos] = {}
					global_dict[chrom][start_pos]['TFs'] = {}
					# global_dict[chrom][start_pos]['idx'] = global_dict_count[chrom]
					global_dict[chrom][start_pos]['bin'] = hic_bin
					global_dict_count[chrom] += 1

				global_dict[chrom][start_pos]['TFs'][TF_id] = 1


	for chrom in global_dict: # sort each chromosome by start position
		global_dict[chrom] = collections.OrderedDict(sorted(global_dict[chrom].items()))
		for idx,start_pos in enumerate(global_dict[chrom]):
			global_dict[chrom][start_pos]['idx'] = idx


	train_idx_dict = {}
	valid_idx_dict = {}
	test_idx_dict = {}

	chrom = 'chr1'
	# chroms = ['chr2']
	total_edges = int(args.size/2.)#125000
	# total_edges = 250000
	# distance_threshold = 
	# chroms = ['chr1']
	for chrom in chroms:
		print(chrom)

		# print('--------Part1-----------')
		# with open('/bigtemp/jjl5sw/hg19/GM12878_combined/1kb_resolution_intrachromosomal/'+chrom+'/MAPQGE30/'+chrom+'_1kb.RAWexpected') as csvfile:
		if args.norm != '':
			normalization_values = []
			with open('/bigtemp/jjl5sw/genome/hg19/'+cell_type+'_combined/'+args.resolution+'kb_resolution_intrachromosomal/'+chrom+'/MAPQGE30/'+chrom+'_'+args.resolution+'kb.'+args.norm+'norm') as csvfile:
				csv_reader = csv.DictReader(csvfile,delimiter='\t',fieldnames=['norm_value'])
				for csv_row in csv_reader:
					norm_val=float(csv_row['norm_value'])
					normalization_values.append(norm_val)
			normalization_values = [1 if math.isnan(x) else x for x in normalization_values]
			normalization_values = [1 if x==0.0 else x for x in normalization_values]

		contact_vals= []
		contact_locs= []
		selected_contact_locs = []

		contact_dict = {}

		# print('--------Part2-----------')
		total_count = 0
		# with open('/bigtemp/jjl5sw/genome/hg19/'+cell_type+'_combined/'+args.resolution+'kb_resolution_intrachromosomal/'+chrom+'/MAPQGE30/'+chrom+'_'+args.resolution+'kb.RAWobserved.sorted') as csvfile:
		with open('/bigtemp/jjl5sw/genome/hg19/hic/'+cell_type+'_combined/'+args.resolution+'kb_resolution_intrachromosomal/'+chrom+'/MAPQGE30/'+chrom+'_'+args.resolution+'kb.RAWobserved.sorted') as csvfile:
			csv_reader = csv.DictReader(csvfile,delimiter='\t',fieldnames=['start_pos1','start_pos2','val'])
			break_flag = False
			for csv_row in tqdm(csv_reader,total=126204768):
				pos1=int(csv_row['start_pos1'])
				pos2=int(csv_row['start_pos2'])
				val = float(csv_row['val'])
				
				if args.norm != '':
					# norm_denom = normalization_values[int((pos2-pos1)/(1000*int(args.resolution)))]
					# val = val/norm_denom

					norm_denom1 = normalization_values[int(int(pos1)/(1000*int(args.resolution)))]
					norm_denom2 = normalization_values[int(int(pos2)/(1000*int(args.resolution)))]

					val = val/(norm_denom1*norm_denom2)


				if cell_type == 'GM12878':
					residuals = [0]
				else:
					residuals = [0,1000,2000,3000,4000]			

				if abs(pos1-pos2) >= args.min_distance_threshold:
					for pos1_add in residuals:
						for pos2_add in residuals:
							pos1_new = pos1+pos1_add
							pos2_new = pos2+pos2_add
							if (pos1_new in global_dict_1000[chrom]) and (pos2_new in global_dict_1000[chrom]):
								contact_vals.append(val)
								contact_locs.append((pos1_new,pos2_new))
								contact_dict[(pos1_new,pos2_new)] = 1#val
								contact_dict[(pos2_new,pos1_new)] = 1#val
								total_count += 1
								


		# print('--------Part3-----------')
		contact_vals=np.array(contact_vals)
		indices = contact_vals.argsort()[-total_edges:][::-1]
		selected_contact_locs = [contact_locs[i] for i in indices]
		selected_contact_locs_dict = {}
		for pos1,pos2 in selected_contact_locs:
			if pos1 not in selected_contact_locs_dict:
				selected_contact_locs_dict[pos1] = [pos1]
			if pos2 not in selected_contact_locs_dict[pos1]:
				selected_contact_locs_dict[pos1].append(pos2)

			if pos2 not in selected_contact_locs_dict:
				selected_contact_locs_dict[pos2] = [pos2]
			if pos1 not in selected_contact_locs_dict[pos2]:
				selected_contact_locs_dict[pos2].append(pos1)

			



		# print('--------Part4-----------')
		adj_mat = np.zeros((len(global_dict[chrom]),len(global_dict[chrom])))#, dtype=np.byte)
		# for chipseq_bin1 in tqdm(global_dict[chrom],leave=False):
		for chipseq_bin1 in global_dict[chrom]:
			hic_bin1 = global_dict[chrom][chipseq_bin1]['bin']
			if hic_bin1 in selected_contact_locs_dict:
				for hic_bin2 in selected_contact_locs_dict[hic_bin1]:
					for res in [0]:#,200,400,600,800]:
						chipseq_bin2 = hic_bin2+res
						if chipseq_bin2 in global_dict[chrom]:
							if chipseq_bin1 != chipseq_bin2:
								bin1idx = global_dict[chrom][chipseq_bin1]['idx']
								bin2idx = global_dict[chrom][chipseq_bin2]['idx']
								val = contact_dict[(hic_bin1,hic_bin2)]
								adj_mat[bin1idx,bin2idx] = val
								adj_mat[bin2idx,bin1idx] = val
			
		
		# print('--------Saving-----------')
		sparse_adj_mat = sparse.csr_matrix(adj_mat)
		if chrom in test_chroms:
			test_idx_dict[chrom] = sparse_adj_mat
		elif chrom in valid_chroms:
			valid_idx_dict[chrom] = sparse_adj_mat
		else:
			train_idx_dict[chrom] = sparse_adj_mat
	

	with open(cell_type+'/test_graphs'+'_'+str(args.size)+'_'+args.norm+'norm.pkl', "wb") as fp: 
		pickle.dump(test_idx_dict, fp)
	with open(cell_type+'/valid_graphs'+'_'+str(args.size)+'_'+args.norm+'norm.pkl', "wb") as fp: 
		pickle.dump(valid_idx_dict, fp)
	with open(cell_type+'/train_graphs'+'_'+str(args.size)+'_'+args.norm+'norm.pkl', "wb") as fp: 
		pickle.dump(train_idx_dict, fp)