#!/usr/bin/env python
import os,sys,csv
import Constants as Constants

hg_file = Constants.hg_file
data_root = Constants.data_root
cell_line=Constants.cell_line
chroms=Constants.chroms
window_length = Constants.window_length
stride_length=Constants.stride_length
extended_window_length=Constants.extended_window_length
genome = Constants.genome

# root=os.path.join(data_root,cell_line,'processed_data')
root = './snp_files/'

sequence_file_name = os.path.join(root,'all_seqs.bed')
output_file_root = os.path.join(root,'fold')


train_file = open(os.path.join(root,'train.txt'),'w')
valid_file = open(os.path.join(root,'valid.txt'),'w')
test_file = open(os.path.join(root,'test.txt'),'w')



valid_chroms = ['chr3', 'chr12', 'chr17']
test_chroms = ['chr1', 'chr8', 'chr21']

print('Input: '+ sequence_file_name)
print('Output: '+ output_file_root)

snp_dict = {}

with open(sequence_file_name) as csvfile:
	csv_reader = csv.DictReader(csvfile,delimiter='\t',fieldnames=['chrom','start_pos','end_pos','ref','alt','fold','label','seq'])
	for csv_row in csv_reader:
		chrom = str(csv_row['chrom']) 
		start_pos = str(csv_row['start_pos'])
		end_pos = str(csv_row['end_pos'])
		ref = str(csv_row['ref'])
		alt = str(csv_row['alt'])
		fold = str(csv_row['fold'])
		label = str(csv_row['label'])
		seq = str(csv_row['seq'])


		if chrom in test_chroms:
			for key,val in csv_row.items():
				test_file.write(val+'\t')
		elif chrom in valid_chroms:
			for key,val in csv_row.items():
				valid_file.write(val+'\t')
		else:
			for key,val in csv_row.items():
				train_file.write(val+'\t')

		

