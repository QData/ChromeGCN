# Process HiChIP data (allValidPairs format) into intra-chromosomal contacts
# Input: line 9 allValidPairs file ('GSM2705062_HCASMC_HiChIP_H3K27ac_B1T1_allValidPairs.txt')
# Output: line 21 row['chr_reads1'].allValidPairs (such as chr1.allValidPairs)
import csv
import numpy as np

#read name / chr_reads1 / pos_reads1 / strand_reads1 / chr_reads2 / pos_reads2 / strand_reads2 / fragment_size [/ allele_specific_tag]
fieldname=['read name','chr_reads1','pos_reads1','strand_reads1','chr_reads2','pos_reads2','strand_reads2','fragment_size','allele_specific_tag']
with open('GSM2705062_HCASMC_HiChIP_H3K27ac_B1T1_allValidPairs.txt') as csvfile:
    reader = csv.DictReader(csvfile,fieldnames=fieldname,delimiter='\t')
    for row in reader:
        if row['chr_reads1']==row['chr_reads2']:
            try:
                row['pos_reads1']=round(int(row['pos_reads1']),-3)
                row['pos_reads2']=round(int(row['pos_reads2']),-3)
                posDiff=np.abs(row['pos_reads2']-row['pos_reads1'])
                row['fragment_size']=posDiff
                cols=['pos_reads1','pos_reads2','fragment_size']
                sublist = [row[x] for x in cols]
                if posDiff>10: # control for longer interaction
                    with open(row['chr_reads1']+'.allValidPairs','a') as output:
                        writer = csv.writer(output,delimiter='\t')
                        writer.writerow(sublist)
                    output.close()
            except IndexError:
                print('index error for ',row['chr_reads1'],'. continuing')
csvfile.close()
