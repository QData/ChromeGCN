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

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-norm', type=str, choices=['','KR','VC','SQRTVC'], default='SQRTVC')
parser.add_argument('-size', type=int, default=250000)
opt = parser.parse_args()


input_root = '/bigtemp/jjl5sw/hg19/K562_combined/5kb_resolution_intrachromosomal'
output_root = '/bigtemp/jjl5sw/hg19/K562_combined/1kb_resolution_intrachromosomal'


for file_path in glob.glob(input_root+'/*'):
    
    chrom = file_path.split('/')[-1]
    print(chrom)
    input_raw_observed = os.path.join(input_root,chrom,'MAPQGE30',chrom+'_5kb.RAWobserved')

    output_chrom_dir = os.path.join(output_root,chrom,'MAPQGE30')
    if not os.path.exists(output_chrom_dir):
        os.makedirs(output_chrom_dir)

    output_raw_observed = open(os.path.join(output_chrom_dir,chrom+'_1kb.RAWobserved'),'w')
    with open(input_raw_observed) as csvfile:
        csv_reader = csv.DictReader(csvfile,delimiter='\t',fieldnames=['start_pos_A','start_pos_B','value'])
        for csv_row in csv_reader:
            start_pos_A = int(csv_row['start_pos_A'])
            start_pos_B = int(csv_row['start_pos_B'])
            value = str(csv_row['value'])
            for res_add_a in [0,1000,2000,3000,4000]:
                for res_add_b in [0,1000,2000,3000,4000]:
                        output_raw_observed.write(str(start_pos_A+res_add_a)+'\t'+str(start_pos_B+res_add_b)+'\t'+value+'\n')
    output_raw_observed.close()
