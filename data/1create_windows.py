'''
This file Generates 'window_length' length windows in hg19
'''

import csv
import math
import os
from pdb import set_trace as stop 



def create_windows(args):
	chrom_sizes = args.chrom_sizes
	output_root = args.output_root
	chroms=args.chroms
	window_length = args.window_length
	extended_window_length = args.extended_window_length
	stride_length=args.stride_length

	print('Inputs')
	input_file_name = chrom_sizes
	print('| '+input_file_name)

	print('\nOutputs')
	windows_file = os.path.join(output_root,'windows.bed')
	extended_windows_file = os.path.join(output_root,'windows_extended.bed')
	print('| '+windows_file)
	print('| '+extended_windows_file)
	print('\n')


	#################################################################

	windows_file_w =  open(windows_file, 'w')
	extended_windows_file_w =  open(extended_windows_file, 'w')

	# import pandas as pd
	# input_pd = pd.read_csv(input_file_name,delimiter='\t',names=['chrom_name','length'])
	# input_pd.set_index('chrom_name').T.to_dict('list')

	lengths = {}
	with open(input_file_name) as csvfile:
		csv_reader = csv.DictReader(csvfile,delimiter='\t',fieldnames=['chrom_name','length'])
		for csv_row in csv_reader:
			lengths[csv_row['chrom_name']] = int(csv_row['length'])-extended_window_length

	for chrom in chroms:
		chrom_length = lengths[chrom]
		prev_i = window_length
		for i in range(window_length,(chrom_length),stride_length):
			start_pos = prev_i
			end_pos = prev_i+window_length
			windows_file_w.write(chrom+'\t'+str(start_pos)+'\t'+str(end_pos)+'\n')

			start_pos_extended = start_pos-int((extended_window_length-window_length)/2)
			end_pos_extended = end_pos+int((extended_window_length-window_length)/2)
			extended_windows_file_w.write(chrom+'\t'+str(start_pos_extended)+'\t'+str(end_pos_extended)+'\n')

			prev_i = i+stride_length


	windows_file_w.close()
	extended_windows_file_w.close()