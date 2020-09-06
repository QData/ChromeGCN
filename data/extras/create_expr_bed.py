
import csv
import os, glob
import Constants as Constants
import numpy as np
from pdb import set_trace as stop 

cell_type = 'E116'

locations_file = '/bigtemp/jjl5sw/DeepChrome/Data/Ensembl_v65.Gencode_v10.ENSG.gene_info'
epr_file = '/bigtemp/jjl5sw/DeepChrome/Data/'+cell_type+'/classification/all_bins.csv'

output_dir = '/bigtemp/jjl5sw/ENCODE/roadmap_expression/'
output_file_name = os.path.join(output_dir,cell_type+'-expr.narrowPeak')

print(epr_file)
print(locations_file)
print(output_file_name)


expr_peak_lengths = []

esng_dict = {}
with open(locations_file) as csvfile:
    csv_reader = csv.DictReader(csvfile,delimiter='\t',fieldnames=['ESNG_id','chrom','start','end'])
    for csv_row in csv_reader:
        ESNG_id = csv_row['ESNG_id']
        chrom = csv_row['chrom']
        start = csv_row['start']
        end = csv_row['end']

        esng_dict[ESNG_id] = {}
        esng_dict[ESNG_id]['chrom'] = 'chr'+chrom
        esng_dict[ESNG_id]['start'] = start
        esng_dict[ESNG_id]['end'] = end

        

output_file = open(output_file_name,'w')
with open(epr_file) as csvfile:
    csv_reader = csv.DictReader(csvfile,delimiter=',',fieldnames=['small_id','bin_id','b1','b2','b3','b4','b5','expr_val'])
    for csv_row in csv_reader:
        small_id = csv_row['small_id']
        expr_val = csv_row['expr_val']

        ESNG_id = 'ENSG00000'+small_id

        if 'expr_val' not in esng_dict[ESNG_id]:
            esng_dict[ESNG_id]['expr_val'] = expr_val
            if int(expr_val) > 0:
                output_file.write(esng_dict[ESNG_id]['chrom']+'\t'+esng_dict[ESNG_id]['start']+'\t'+esng_dict[ESNG_id]['end']+'\t'+ESNG_id+'\t'+esng_dict[ESNG_id]['expr_val']+'\n')
            
                expr_peak_lengths.append(int(esng_dict[ESNG_id]['end'])-int(esng_dict[ESNG_id]['start']))

expr_peak_lengths = np.array(expr_peak_lengths)
print('mean:   ',np.mean(expr_peak_lengths))
print('median: ',np.median(expr_peak_lengths))
print('std:    ',np.std(expr_peak_lengths))
print('max:    ',np.max(expr_peak_lengths))
print('min:    ',np.min(expr_peak_lengths))
            
output_file.close()

