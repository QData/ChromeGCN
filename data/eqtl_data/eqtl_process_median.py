# Process eQTL data by thresholding based on median expression value
# Input: line 15 ('GSE113348_rnaseqc.hcasmc_eqtl.reads.gct')
# Output: line 32 'processed_hcasmc_eqtl_median.reads'
import csv
import numpy as np


header=0 # for skipping header lines
names=[] # list of ENSG
values=[] #list of all averaged values matching the list of names

#read name / note / samples (52 columns)   [total 56238 rows]
with open('GSE113348_rnaseqc.hcasmc_eqtl.reads.gct') as csvfile:
    reader = csv.reader(csvfile,delimiter='\t')
    for row in reader:
        if header!=3: #skip 2 lines of header
            header+=1
            continue
        names.append(row[0])
        #sum(int(row[i]) for i in range(2,len(row)))
        value=sum(int(row[i]) for i in range(2,len(row)))/(len(row)-2) #average of all samples of this row
        values.append(value)

median_expr=np.median(values) #median expression level for thresholding

#count1=0
#count0=0
with open('processed_hcasmc_eqtl_median.reads','a') as csvfile:
    writer = csv.writer(csvfile,delimiter='\t')
    for i in range(len(names)):
        if (values[i]>=median_expr):
            writer.writerow([names[i],1])
            #count1+=1
        else:
            writer.writerow([names[i],0])
            #count0+=1
#print('1: ',count1,". 0: ",count0)
