import csv
import numpy as np


header=0 # for skipping header lines
total=0 # for mean expression calculation
count=0 # count number of rows
names=[] # list of ENSG
values=[] #list of all averaged values matching the list of names

#read name / note / samples (52 columns)   [total 56238 rows]
with open('GSE113348_rnaseqc.hcasmc_eqtl.reads.gct') as csvfile:
    reader = csv.reader(csvfile,delimiter='\t')
    for row in reader:
        if header!=3: #skip 2 lines of header
            header+=1
            continue
        count+=1
        names.append(row[0])
        #sum(int(row[i]) for i in range(2,len(row)))
        value=sum(int(row[i]) for i in range(2,len(row)))/(len(row)-2) #average of all samples of this row
        values.append(value)
        total+=value

mean_expr=total/count #mean expression level for thresholding
exit()
with open('processed_hcasmc_eqtl.reads','a') as csvfile:
    writer = csv.writer(csvfile,delimiter='\t')
    for i in range(len(names)):
        if (values[i]>=mean_expr):
            writer.writerow([names[i],1])
        else:
            writer.writerow([names[i],0])

