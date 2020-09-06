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
windows_file_name = os.path.join(root,'all_windows.bed')
sequence_file_name = os.path.join(root,'all_seqs.bed')
fasta_file_name= os.path.join(root,'all_seqs.fa')

print('Input: '+ windows_file_name)
print('Temp: '+ fasta_file_name)
print('Output2: '+ sequence_file_name)


snp_dict = {}

with open(windows_file_name) as csvfile:
	csv_reader = csv.DictReader(csvfile,delimiter='\t',fieldnames=['chrom','start_pos','end_pos','ref','alt','fold','label'])
	for csv_row in csv_reader:
		chrom = str(csv_row['chrom']) 
		start_pos = str(csv_row['start_pos'])
		end_pos = str(csv_row['end_pos'])
		ref = str(csv_row['ref'])
		alt = str(csv_row['alt'])
		fold = str(csv_row['fold'])
		label = str(csv_row['label'])

		snp_dict[start_pos] =csv_row
		


def create_fa():
	# cmd = 'bedtools getfasta -fi /bigtemp/jjl5sw/hg38/hg38.fa -bed '+windows_file_name+' -fo '+fasta_file_name
	cmd = 'bedtools getfasta -fi /bigtemp/jjl5sw/'+genome+'/'+genome+'.fa -bed '+windows_file_name+' -fo '+fasta_file_name
	os.system(cmd)


def create_bed_seqs():
	bed_seqs_file = open(sequence_file_name,'w')

	f = open(fasta_file_name,'r')

	for line in f:
		if '>' in line:
			chrom = line.split('>')[1].split(':')[0]
			start_pos = int(line.split(':')[1].split('-')[0])
			end_pos = int(line.split(':')[1].split('-')[1])
		else:
			csv_row = snp_dict[str(start_pos)]
			chrom = str(csv_row['chrom']) 
			start_pos = str(csv_row['start_pos'])
			end_pos = str(csv_row['end_pos'])
			ref = str(csv_row['ref'])
			alt = str(csv_row['alt'])
			fold = str(csv_row['fold'])
			label = str(csv_row['label'])
            if ref != line.upper()[1000]:
                print('Error')
			bed_seqs_file.write(chrom+'\t'+str(start_pos)+'\t'+str(end_pos)+'\t'+str(ref)+'\t'+str(alt)+'\t'+str(fold)+'\t'+str(label)+'\t'+line.upper())

	f.close()
	bed_seqs_file.close()

	# os.system('rm '+fasta_file_name)



print('create_fa')
create_fa()

print('create_bed_seqs')
create_bed_seqs()
