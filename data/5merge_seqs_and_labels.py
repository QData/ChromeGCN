#!/usr/bin/env python

'''
Creates single file for each window which includes the sequence and the 
binary label for each chip-seq signal
'''

import csv,math,os, glob,sys,pickle,collections
from random import randint, choice
from pdb import set_trace as stop

def merge_seqs_and_labels(args):
	chroms=args.chroms
	window_length = args.window_length
	extended_window_length = args.extended_window_length
	input_root = args.input_root
	output_root = args.output_root

	####################################################################################
	print('Inputs')
	all_peaks_file_name = os.path.join(output_root,'chipseq_windows_extended.bed')
	if args.use_all_windows:
		all_seqs_file_name = os.path.join(output_root,'windows_extended.seq')
	else:
		all_seqs_file_name = os.path.join(output_root,'chipseq_windows_extended.seq')
	print('| '+all_peaks_file_name)
	print('| '+all_seqs_file_name)

	print('\nOutputs')
	output_file_name = os.path.join(output_root,'windows_and_pvalues.txt')
	print('| '+output_file_name)
	print('\n')

	####################################################################################
	print('=> Create Window Sequence Dict')

	bin_dict={chrom:{} for chrom in chroms}  #Format: bin_dict[chrom][start_pos]['labels'][label] = pvalue

	with open(all_seqs_file_name) as csvfile:
		csv_reader = csv.DictReader(csvfile,delimiter='\t',fieldnames=['chrom','start_pos','end_pos','SEQ'])
		for line in csv_reader:
			chrom = line['chrom']
			start_pos = int(line['start_pos'])
			seq = line['SEQ']

			bin_dict[chrom][start_pos] = {}
			bin_dict[chrom][start_pos]['seq'] = seq
			bin_dict[chrom][start_pos]['labels'] = {}

	####################################################################################
	print('=> Create ChIP-Seq Label Dict')

	chipseq_ids = []
	with open(all_peaks_file_name) as csvfile:
		csv_reader = csv.DictReader(csvfile,delimiter='\t',fieldnames=['chrom','start_pos','end_pos','label','score','strand','signalValue','pvalue','qValue','peak'])
		for csv_row in csv_reader:
			label = str(csv_row['label'])
			chrom = str(csv_row['chrom'])
			start_pos = int(csv_row['start_pos'])
			# pvalue = float(csv_row['pvalue'])

			if not label in chipseq_ids:
				chipseq_ids.append(label)

			# bin_dict[chrom][start_pos]['labels'][label] = pvalue
			bin_dict[chrom][start_pos]['labels'][label] = 1

	# sort each chromosome by start position
	for chrom in bin_dict:
		bin_dict[chrom] = collections.OrderedDict(sorted(bin_dict[chrom].items()))


	####################################################################################
	print('=> Write Sequences and Labels to Output File')
	output_file =  open(output_file_name, 'w')
	output_file.write('chrom\tstart_pos\tend_pos\tSEQ')
	for label in chipseq_ids: output_file.write('\t'+str(label))
	output_file.write('\n')

	difference = int((extended_window_length-window_length)/2)

	for chrom in chroms:
		for start_pos in bin_dict[chrom]:
			seq = bin_dict[chrom][start_pos]['seq']

			output_file.write(str(chrom)+'\t'+str(start_pos+difference)+'\t'+str(start_pos+window_length+difference))
			output_file.write('\t'+seq.upper())
			for label in chipseq_ids:
				if label in bin_dict[chrom][start_pos]['labels']:
					output_file.write('\t'+str(bin_dict[chrom][start_pos]['labels'][label]))
				else:
					output_file.write('\t0')
			output_file.write('\n')
				
			
	output_file.close()


