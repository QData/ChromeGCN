#!/usr/bin/env python

# creates a single file for all TFs. 'name' column in the narrowPeak bed file
# is the TF ID

import csv
import os, glob
import Constants as Constants
import numpy as np
from pdb import set_trace as stop 
import math
# import sys
# from random import randint, choice
# import collections


data_root = Constants.data_root
cell_line=Constants.cell_line
chroms=Constants.chroms
window_length = Constants.window_length


cell_line_dir = data_root+cell_line

window_length_half = int(float(window_length)/2)

output_directory = './snp_files/'

input_file_name = 'grasp_eqtl_snps.csv'
output_file_name = os.path.join(output_directory,'all_windows.bed')

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

print('Input: '+input_file_name)
print('Output: '+output_file_name)

output_file =  open(output_file_name, 'w')

all_seq_lengths = [] 


# print(label_name)
with open(input_file_name) as csvfile:
    csv_reader = csv.DictReader(csvfile,delimiter=',',fieldnames=['idx','chrom','pos','ref','alt','probability','fold','label'])
    sample_idx = 0
    for sample in csv_reader:
        sample_idx+=1
        if sample_idx > 4:
            # print(sample)
            idx = sample['idx']
            chrom = sample['chrom']
            pos = sample['pos']
            ref = sample['ref']
            alt = sample['alt']
            fold = sample['fold']
            label = sample['label']


            start_pos = max(0,int(pos)-1001)
            end_pos = int(pos)+999

            output_file.write(str(chrom)+'\t'+str(start_pos)+'\t'+str(end_pos)+'\t'+ref+'\t'+alt+'\t'+fold+'\t'+label+'\n')
            # output_file.write(chrom+'\t'+str(start_pos)+'\t'+str(end_pos)+'\t'+label_name+'\t'+score+'\t'+strand+'\t'+signalValue+'\t'+pvalue+'\t'+qValue+'\t'+str(peak)+'\n')
		



output_file.close()