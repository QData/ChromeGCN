#!/usr/bin/env python

'''
Maps each peak of variable lenghts to the predefined windows from 1create_windows
'''

import csv,os,glob
import tempfile
from pdb import set_trace as stop


def create_windows_with_peaks(args):
    chroms=args.chroms
    window_length = args.window_length
    extended_window_length=args.extended_window_length
    output_root = args.output_root

    print('Inputs')
    windows_file = os.path.join(output_root,'windows.bed')
    chipseq_peaks_file=os.path.join(output_root,'chipseq_peaks.bed')
    print('| '+windows_file)
    print('| '+chipseq_peaks_file)

    temp_file=os.path.join(output_root,'chipseq_windows.tmp')

    print('\nOutputs')
    chipseq_windows_file=os.path.join(output_root,'chipseq_windows.bed')
    extended_chipseq_windows_file = os.path.join(output_root,'chipseq_windows_extended.bed')
    print('| '+chipseq_windows_file)
    print('| '+extended_chipseq_windows_file)
    # print('| '+extended_windows_file)
    print('\n')

    #################################################################
    

    method = 'A'

    if method == 'A':
        # Any intersection over 50% of windows surrounding peak
        # -f flag: Minimum overlap required as a fraction of -a
        # -F flag: Minimum overlap required as a fraction of -b
        sys_command='bedtools intersect -wa -wb -f 0.1 -a '+windows_file+' -b  '+chipseq_peaks_file+' > '+temp_file
    elif method == 'B':
        sys_command='bedtools intersect -wa -wb -f 0.5 -a '+all_peaks_file+' -b  '+bed_windows_file+' > '+temp_file

    os.system(sys_command)


    # sys_command2 = '''awk \'{print $11\"\\t\"$12\"\\t\"$13\"\\t\"$4\"\\t\"$5\"\\t\"$6\"\\t\"$7\"\\t\"$8\"\\t\"$9\"\\t\"$10}\' '''+temp_file+' > '+chipseq_windows_file
    sys_command2 = '''awk \'{print $1\"\\t\"$2\"\\t\"$3\"\\t\"$7\"\\t\"$8\"\\t\"$9\"\\t\"$10\"\\t\"$11\"\\t\"$12\"\\t\"$13}\' '''+temp_file+' > '+chipseq_windows_file
    os.system(sys_command2)

    sys_command3='rm '+temp_file
    os.system(sys_command3)


    ## Create Extended Windows (surrounding original ones) ##
    # difference = int((extended_window_length-window_length)/2)
    # sys_command = "awk -F ',' '{$2-="+str(difference)+"}{$3+="+str(difference)+"}1' " + windows_file_name + " > " + extended_windows_file
    # os.system(sys_command)
    with open(extended_chipseq_windows_file,'w') as output_file:
        with open(chipseq_windows_file) as csvfile:
            csv_reader = csv.DictReader(csvfile,delimiter='\t',fieldnames=['chrom','start_pos','end_pos','name','score','strand','signalValue','pvalue','qValue','peak'])
            for sample in csv_reader:
                chrom = sample['chrom']
                start_pos = sample['start_pos']
                end_pos = sample['end_pos']
                score = sample['score']
                strand = sample['strand']
                signalValue = sample['signalValue']
                pvalue = sample['pvalue']
                qValue = sample['qValue']
                peak = sample['peak']
                TF_id = sample['name']

                try:
                    peak = int(peak)
                except ValueError:
                    peak = 0

                peak_pos = int(start_pos)+peak

                # 1000 length around the center of each window
                start_pos = int(start_pos)-int((extended_window_length-window_length)/2)
                end_pos = int(end_pos)+int((extended_window_length-window_length)/2)

                output_file.write(chrom+'\t'+str(start_pos)+'\t'+str(end_pos)+'\t'+TF_id+'\t'+score+'\t'+strand+'\t'+signalValue+'\t'+pvalue+'\t'+qValue+'\t'+str(peak)+'\n')


