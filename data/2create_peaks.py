#!/usr/bin/env python

'''
Creates a single bed file containing all assays. 'name' column in the narrowPeak bed file is the assay ID
'''

import csv
import os, glob
import numpy as np
from pdb import set_trace as stop 

def create_peaks(args):
	chroms_to_use=args.chroms
	cell_type = args.cell_type

	window_length_half = int(float(args.window_length)/2)

	print('Inputs')
	input_file_name = os.path.join(args.input_root+'*')
	print('| '+input_file_name)
	if args.expr_root != '':
		expression_input_dir = os.path.join(expr_root+'*')
		print('| '+expression_input_dir)
	else:
		expression_input_dir = None

	print('\nOutputs')
	output_file_name = os.path.join(args.output_root,'chipseq_peaks.bed')
	print('| '+output_file_name)
	print('\n')

	#################################################################

	tf_peak_lengths = []
	hm_peak_lengths = []
	dnase_peak_lengths = []
	expr_peak_lengths = []
	all_peak_lengths = []

	if cell_type == 'GM12878':
		roadmap_cell_type = 'E116'
	elif cell_type == 'K562':
		roadmap_cell_type = 'E123'
	else:
		roadmap_cell_type = ''

	output_file =  open(output_file_name, 'w')
	for file_path in glob.glob(input_file_name):
		if (cell_type.lower() in file_path.lower()) or (roadmap_cell_type.lower() in file_path.lower()):
			file_dir, file_name = os.path.split(file_path)
			label_name = file_name.split('.')[0]
			with open(file_path) as csvfile:
				csv_reader = csv.DictReader(csvfile,delimiter='\t',fieldnames=['chrom','start_pos','end_pos','name','score','strand','signalValue','pvalue','qValue','peak'])
				for sample in csv_reader:
					chrom = sample['chrom']
					start_pos = sample['start_pos']
					end_pos = sample['end_pos']
					score = sample['score']
					strand = sample['strand']
					signalValue = sample['signalValue']
					pvalue = sample['pvalue']
					qValue = sample['qValue']
					peak = sample['peak']

					if chrom in chroms_to_use:
						## METHOD A ##
						# peak_pos = int(start_pos)+int(peak)
						# output_file.write(chrom+'\t'+str(peak_pos-window_length_half)+'\t'+str(peak_pos+window_length_half)+'\t'+label_name+'\t'+score+'\t'+strand+'\t'+signalValue+'\t'+pvalue+'\t'+qValue+'\t'+str(window_length_half)+'\n')

						## METHOD B ##
						output_file.write(chrom+'\t'+str(start_pos)+'\t'+str(end_pos)+'\t'+label_name+'\t'+score+'\t'+strand+'\t'+signalValue+'\t'+pvalue+'\t'+qValue+'\t'+str(peak)+'\n')
						all_peak_lengths.append(int(end_pos)-int(start_pos))
						if 'Tfbs'in file_path:
							tf_peak_lengths.append(int(end_pos)-int(start_pos))
						elif 'Dnase' in file_path:
							dnase_peak_lengths.append(int(end_pos)-int(start_pos))
						else:
							hm_peak_lengths.append(int(end_pos)-int(start_pos))

	## Expression ##
	if expression_input_dir is not None:
		for file_path in glob.glob(expression_input_dir):
			if (cell_type.lower() in file_path.lower()) or (roadmap_cell_type.lower() in file_path.lower()):
				file_dir, file_name = os.path.split(file_path)
				label_name = file_name.split('.')[0]
				with open(file_path) as csvfile:
					csv_reader = csv.DictReader(csvfile,delimiter='\t',fieldnames=['chrom','start_pos','end_pos','name','val'])
					for sample in csv_reader:
						chrom = sample['chrom']
						start_pos = sample['start_pos']
						end_pos = sample['end_pos']
						score = sample['val']

						if chrom in chroms_to_use:
							output_file.write(chrom+'\t'+str(start_pos)+'\t'+str(end_pos)+'\t'+label_name+'\t'+score+'\t'+'-'+'\t'+'0'+'\t'+'0'+'\t'+'0'+'\t'+str('0')+'\n')
							expr_peak_lengths.append(int(end_pos)-int(start_pos))



	output_file.close()

	print_stats = False
	if print_stats:
		all_peak_lengths = np.array(all_peak_lengths)
		tf_peak_lengths = np.array(tf_peak_lengths)
		hm_peak_lengths = np.array(hm_peak_lengths)
		dnase_peak_lengths = np.array(dnase_peak_lengths)
		expr_peak_lengths = np.array(expr_peak_lengths)

		print('\nALL')
		print('mean:   ',np.mean(all_peak_lengths))
		print('median: ',np.median(all_peak_lengths))
		print('std:    ',np.std(all_peak_lengths))
		print('max:    ',np.max(all_peak_lengths))
		print('min:    ',np.min(all_peak_lengths))

		try:
			print('\nTFBS')
			print('mean:   ',np.mean(tf_peak_lengths))
			print('median: ',np.median(tf_peak_lengths))
			print('std:    ',np.std(tf_peak_lengths))
			print('max:    ',np.max(tf_peak_lengths))
			print('min:    ',np.min(tf_peak_lengths))

			print('\nHM')
			print('mean:   ',np.mean(hm_peak_lengths))
			print('median: ',np.median(hm_peak_lengths))
			print('std:    ',np.std(hm_peak_lengths))
			print('max:    ',np.max(hm_peak_lengths))
			print('min:    ',np.min(hm_peak_lengths))

			print('\nDnase')
			print('mean:   ',np.mean(dnase_peak_lengths))
			print('median: ',np.median(dnase_peak_lengths))
			print('std:    ',np.std(dnase_peak_lengths))
			print('max:    ',np.max(dnase_peak_lengths))
			print('min:    ',np.min(dnase_peak_lengths))

			# stop()
			print('\nExpr')
			print('mean:   ',np.mean(expr_peak_lengths))
			print('median: ',np.median(expr_peak_lengths))
			print('std:    ',np.std(expr_peak_lengths))
			print('max:    ',np.max(expr_peak_lengths))
			print('min:    ',np.min(expr_peak_lengths))
		except:
			pass

		all_peak_lengths[::-1].sort()