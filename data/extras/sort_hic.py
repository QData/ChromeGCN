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
parser.add_argument('-cell_type', type=str, choices=['GM12878','K562'], default='GM12878')
opt = parser.parse_args()



if opt.cell_type == 'GM12878':
    resolution = '1'
else:
    resolution = '5'


input_root = '/bigtemp/jjl5sw/hg19/'+opt.cell_type+'_combined/'+resolution+'kb_resolution_intrachromosomal'

print(input_root)

for file_path in glob.glob(input_root+'/*'):
    chrom = file_path.split('/')[-1]
    input_raw_observed = os.path.join(input_root,chrom,'MAPQGE30',chrom+'_'+resolution+'kb.RAWobserved')
    output_raw_observed_sorted = os.path.join(input_root,chrom,'MAPQGE30',chrom+'_'+resolution+'kb.RAWobserved.sorted')

    cmd = 'sort -r -k3 -n '+input_raw_observed+' > '+output_raw_observed_sorted
    print(cmd)
    os.system(cmd)

