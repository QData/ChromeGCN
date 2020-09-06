# Convert pre-processed eQTL data by replacing ENSG with TSS-TES location pairs of all transcripts
# Require: 'hg19.ensGene.gtf' as dictionary; obtained from http://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/genes/
# Input: line 41 'processed_hcasmc_rpkm.txt'
# Output: line 51 'hcasmc_rpkm_train.txt'
import csv
import numpy as np
import os
import sys


csv.field_size_limit(sys.maxsize)
dictionary=dict()
with open('hg19.ensGene.gtf') as csvfile:
    reader = csv.reader(csvfile,delimiter='\t')
    for row in reader:
        name=(row[8].split(';')[0].split('"')[1]) #extract ENSG from row
        if name not in dictionary:
            dictionary[name]=[]
        #create traning BED file: row[0]=chr#, row[3]=TSS, row[4]=TES
        #strings as place holders to match BED format of other data files
        dictionary[name].append([row[0],row[3],row[4],'Peak_Name','score','strand','signalValue','pvalue','qvalue','peak'])
csvfile.close()

#saving dictionary to 'dictionary.txt'
with open('dictionary.txt','a') as csvfile:
    writer = csv.writer(csvfile,delimiter='\t')
    for i in dictionary:
        writer.writerow([i,dictionary[i]])
        #strings as place holders to match BED format of other data files
csvfile.close()


'''
names=[] # list of ENSG
variants=[] #list of eQTL variants 
'''
genes=[]
#read name / note / samples (52 columns)   [total 56238 rows]
with open('processed_hcasmc_rpkm.txt') as csvfile:
    reader = csv.reader(csvfile,delimiter='\t')
    for row in reader:
        if int(row[1])==1:#keeping only binary==1
            name=row[0].split('.')[0]
            if name in dictionary:
                genes.append(dictionary[name])
csvfile.close()

print(len(genes))
with open('hcasmc_rpkm_train.txt','a') as train:
    writer = csv.writer(train,delimiter='\t')
    for gene in genes:
        for transcript in gene:
            writer.writerow(transcript)
train.close()
'''
    #create testing BED file: chr#, TSS, TES, list of all eQTL variants
    #keeping only binary==1
    with open('hcasmc_rpkm_test.txt','a') as test:
        writer = csv.writer(test,delimiter='\t')
        writer.writerow([row[i],r_variants[i]])
    test.close()
'''




