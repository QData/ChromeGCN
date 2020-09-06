#!/usr/bin/env python

'''
Gets the sequence data for each window
'''

import os,sys,csv

def create_seqs(args):
    genome_fasta = args.genome_fasta
    window_length = args.window_length
    extended_window_length=args.extended_window_length
    output_root = args.output_root

    #################################################################
    print('Inputs')
    if args.use_all_windows:
        extended_windows_file_name = os.path.join(output_root,'windows_extended.bed') 
    else:
        extended_windows_file_name = os.path.join(output_root,'chipseq_windows_extended.bed') 
    print('| '+ extended_windows_file_name)

    print('\nOutputs')
    if args.use_all_windows:
        sequence_file_name = os.path.join(output_root,'windows_extended.seq')
    else:
        sequence_file_name = os.path.join(output_root,'chipseq_windows_extended.seq')
    print('| '+ sequence_file_name)
    print('\n')
    #################################################################


    # Get sequences from specified windows in extended_windows_file_name
    cmd = 'bedtools getfasta -tab -fi '+genome_fasta+' -bed '+extended_windows_file_name+' -fo '+sequence_file_name
    os.system(cmd)

    # Replace ":" and "-" with tab
    cmd = "sed -i 's/:/\\t/g; s/-/\\t/g' "+ sequence_file_name
    os.system(cmd)

