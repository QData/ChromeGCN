#!/usr/bin/env python

import os,sys,csv,glob,pickle,random
import torch
from multiprocessing import Process
import os.path

from pdb import set_trace as stop

def create_input_label_files(args):
	chroms=args.chroms
	output_root = args.output_root

	valid_chroms = args.valid_chroms
	test_chroms = args.test_chroms

	alphabet = ['A','C','G','T','N']

	input_file = os.path.join(output_root,'windows_and_pvalues.txt')

	output_root = args.output_root 

	print('Inputs')
	print('| '+input_file)
	print('\nOutputs')
	print('| '+output_root+'/*')
	print('\n')

	train_locs = open(os.path.join(output_root,'train_locs.txt'),'w')
	train_inputs = open(os.path.join(output_root,'train_inputs.txt'),'w')
	train_labels= open(os.path.join(output_root,'train_labels.txt'),'w')

	valid_locs = open(os.path.join(output_root,'valid_locs.txt'),'w')
	valid_inputs = open(os.path.join(output_root,'valid_inputs.txt'),'w')
	valid_labels= open(os.path.join(output_root,'valid_labels.txt'),'w')

	test_locs = open(os.path.join(output_root,'test_locs.txt'),'w')
	test_inputs = open(os.path.join(output_root,'test_inputs.txt'),'w')
	test_labels= open(os.path.join(output_root,'test_labels.txt'),'w')

	with open(input_file, "r") as csvfile:
		headers = next(csvfile).split()
		label_names = headers[4:]
		TF_lists_dict = {}
		for TF in label_names:
			TF_lists_dict[TF] = []
		seqs_dict={chrom:{} for chrom in chroms} 
		SEQ_list = []
		csv_reader = csv.DictReader(csvfile,delimiter='\t',fieldnames=headers)

		train_idxs = []
		valid_idxs = []
		test_idxs = []

		line_idx = 0
		for line in csv_reader:
			line_idx +=1
			chrom = line['chrom']
			start = line['start_pos']
			end = line['end_pos']
			seq = line['SEQ']
			split_seq = [char for char in seq]
			split_seq = " ".join(map(str, split_seq))

			if chrom in test_chroms:
				test_idxs.append(chrom)
				test_locs.write(chrom+'\t'+start+'\t'+end+'\n')
				test_inputs.write(split_seq+'\n')
				for TF in label_names:
					pvalue = float(line[TF])
					if pvalue > 0.0:
						test_labels.write(TF+' ')
				test_labels.write('\n')
			elif chrom in valid_chroms:
				valid_idxs.append(chrom)
				valid_locs.write(chrom+'\t'+start+'\t'+end+'\n')
				valid_inputs.write(split_seq+'\n')
				for TF in label_names:
					pvalue = float(line[TF])
					if pvalue > 0.0:
						valid_labels.write(TF+' ')
				valid_labels.write('\n')
			else:
				train_idxs.append(chrom)
				train_locs.write(chrom+'\t'+start+'\t'+end+'\n')
				train_inputs.write(split_seq+'\n')
				for TF in label_names:
					pvalue = float(line[TF])
					if pvalue > 0.0:
						train_labels.write(TF+' ')
				train_labels.write('\n')

	train_locs.close()
	train_inputs.close()
	train_labels.close()

	valid_locs.close()
	valid_inputs.close()
	valid_labels.close()

	test_locs.close()
	test_inputs.close()
	test_labels.close()

	# with open(os.path.join(output_root,"train_idx_list.pkl"), "wb") as fp: 
	# 	pickle.dump(train_idxs, fp)
	# with open(os.path.join(output_root,"valid_idx_list.pkl"), "wb") as fp: 
	# 	pickle.dump(valid_idxs, fp)
	# with open(os.path.join(output_root,"test_idx_list.pkl"), "wb") as fp: 
	# 	pickle.dump(test_idxs, fp)

