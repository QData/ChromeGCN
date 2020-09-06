# Process eQTL data by thresholding based on median expression value and attach eQTL variants to each gene 
# Input: line 20 'hcasmc_rpkm.txt' and line 40 'chr'+str(f)) (variants by chromosome)
# Output: line 57 'processed_hcasmc_rpkm.txt'
import csv
import numpy as np
import os

'''
transfer data:

rsync --rsh='ssh' -a --progress --partial eQTL ew8sk@portal.cs.virginia.edu:/p/qdata_genomics/ew8sk/CAD_DATA
'''


header=0 # for skipping header lines
names=[] # list of ENSG
values=[] #list of all averaged values matching the list of names

#read name / note / samples (52 columns)   [total 56238 rows]
with open('hcasmc_rpkm.txt') as csvfile:
    reader = csv.reader(csvfile,delimiter='\t')
    for row in reader:
        if header!=2: #skip 1 line of header
            header+=1
            continue
        names.append(row[0])
        #sum(int(row[i]) for i in range(2,len(row)))
        value=sum(float(row[i]) for i in range(2,len(row)))/(len(row)-2) #average of all samples of this row
        values.append(value)

median_expr=np.median(values) #median expression level for thresholding
csvfile.close()

#count1=0
#count0=0

script_dir = os.path.dirname(__file__)

for f in range(1,23):
    abs_file_path = os.path.join(script_dir, 'chr'+str(f))
    for filename in os.listdir(abs_file_path):
        name=filename.split('_')[0]
        #print(name)
        try:
            value=values[names.index(name)]
        except ValueError:
            continue
        #print(value)
        variants=[]
        path=os.path.join(abs_file_path,filename)
        with open(path) as csvfile:
            reader = csv.reader(csvfile,delimiter='\t')
            for row in reader:
                variants.append(row)
        csvfile.close()

        with open('processed_hcasmc_rpkm.txt','a') as csvfile:
            writer = csv.writer(csvfile,delimiter='\t')
            if (value>=median_expr):
                writer.writerow([name,1,variants])
                #count1+=1
            else:
                writer.writerow([name,0,variants])
                #count0+=1
        #print('1: ',count1,". 0: ",count0)
        csvfile.close()



