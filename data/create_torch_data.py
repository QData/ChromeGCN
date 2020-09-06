
import argparse
import torch
import Constants
from pdb import set_trace as stop
from scipy import sparse
import operator, collections
import os

"""
Adapted from https://github.com/jadore801120/attention-is-all-you-need-pytorch
"""

def read_instances_from_file(inst_file, max_sent_len, keep_case, use_bos_eos=True):
    ''' Convert file into word seq lists and vocab '''

    word_insts = []
    trimmed_sent_count = 0
    with open(inst_file) as f:
        for sent in f:
            if not keep_case:
                sent = sent.lower()
            word_inst = sent.split()
            if max_sent_len != -1:
                if len(word_inst) > max_sent_len:
                    trimmed_sent_count += 1
                word_inst = word_inst[:max_sent_len]

            if word_inst:
                if use_bos_eos:
                    word_insts += [[Constants.BOS_WORD] + word_inst + [Constants.EOS_WORD]]
                else:
                    word_insts += [word_inst]
            else:
                word_insts += ['']


    if trimmed_sent_count > 0:
        print('[Warning] {} instances are trimmed to the max sentence length {}.'
              .format(trimmed_sent_count, max_sent_len))

    return word_insts

def build_vocab_idx(word_insts, min_word_count,use_bos_eos=True):
    ''' Trim vocab by number of occurence '''

    try:
        full_vocab = set(w for sent in word_insts for w in sent)
    except:
        stop()
    print('[Info] Original Vocabulary size =', len(full_vocab))

    if use_bos_eos:
        word2idx = {
            Constants.BOS_WORD: Constants.BOS,
            Constants.EOS_WORD: Constants.EOS,
            Constants.PAD_WORD: Constants.PAD,
            Constants.UNK_WORD: Constants.UNK}
    else:
        word2idx = {}

    word_count = {w: 0 for w in full_vocab}

    for sent in word_insts:
        for word in sent:
            word_count[word] += 1

    ignored_word_count = 0

    word_count = sorted(word_count.items(), key=operator.itemgetter(1),reverse=True)
    word_count = collections.OrderedDict(word_count)
    for word, count in word_count.items():
        if word not in word2idx:
            if count > min_word_count:
                word2idx[word] = len(word2idx)
            else:
                ignored_word_count += 1

    print('[Info] Trimmed vocabulary size = {},'.format(len(word2idx)),
          'each with minimum occurrence = {}'.format(min_word_count))
    print("[Info] Ignored word count = {}".format(ignored_word_count))
    return word2idx

def convert_instance_to_idx_seq(word_insts, word2idx):
    '''Word mapping to idx'''
    word_insts = ['</s>' if x is None else x for x in word_insts]
    return [[word2idx[w] for w in s] for s in word_insts]
    # return [[word2idx[w] if w in word2idx else Constants.UNK for w in s] for s in word_insts]

def convert_instance_to_binary_vec(word_insts, word2idx):
    binary_vecs = []
    # stop()
    for sample in word_insts:
        binary_vec = [0 for i in range(len(word2idx))]
        for label in sample:
            binary_vec[word2idx[label]] = 1
        binary_vecs.append(binary_vec)

    return binary_vecs

def divide_into_chroms(loc_insts,src_insts,tgt_insts):
    split_chrom_dict = {}
    for idx in range(len(loc_insts)):
        loc = loc_insts[idx]
        src = src_insts[idx]
        tgt = tgt_insts[idx]
        
        # chrom = loc[0] 
        chrom = loc_insts[0][0]
        if not chrom in split_chrom_dict:
            split_chrom_dict[chrom] = {'loc':[],'src':[],'tgt':[]}
        
        split_chrom_dict[chrom]['loc'].append(loc)
        split_chrom_dict[chrom]['src'].append(src)
        split_chrom_dict[chrom]['tgt'].append(tgt)

    return split_chrom_dict
        

def main():
    ''' Main function '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-train_locs', default='train_locs.txt')
    parser.add_argument('-train_src', default='train_inputs.txt')
    parser.add_argument('-train_tgt', default='train_labels.txt')
    parser.add_argument('-valid_src', default='valid_inputs.txt')
    parser.add_argument('-valid_locs', default='valid_locs.txt')
    parser.add_argument('-valid_tgt', default='valid_labels.txt')
    parser.add_argument('-test_src', default='test_inputs.txt')
    parser.add_argument('-test_locs', default='test_locs.txt')
    parser.add_argument('-test_tgt', default='test_labels.txt')
    parser.add_argument('-save_data', default='train_valid_test.pt')
    # parser.add_argument('-train_locs', default='train_locs_small.txt')
    # parser.add_argument('-train_src', default='train_inputs_small.txt')
    # parser.add_argument('-train_tgt', default='train_labels_small.txt')
    # parser.add_argument('-valid_src', default='valid_inputs_small.txt')
    # parser.add_argument('-valid_locs', default='valid_locs_small.txt')
    # parser.add_argument('-valid_tgt', default='valid_labels_small.txt')
    # parser.add_argument('-test_src', default='test_inputs_small.txt')
    # parser.add_argument('-test_locs', default='test_locs_small.txt')
    # parser.add_argument('-test_tgt', default='test_labels_small.txt')
    # parser.add_argument('-save_data', default='train_valid_test_small.pt')
    parser.add_argument('-data_root', default='/p/qdata/jjl5sw/ChromeGCN/data/GM12878/1000/')
    parser.add_argument('-vec_inputs', action='store_true')
    parser.add_argument('-divide_split_into_chroms', action='store_true')
    parser.add_argument('-max_seq_len', type=int, default=-1)
    parser.add_argument('-max_tgt_len', type=int, default=-1)
    parser.add_argument('-min_word_count', type=int, default=0)
    parser.add_argument('-keep_case', action='store_true')
    parser.add_argument('-share_vocab', action='store_true')
    opt = parser.parse_args()

    # opt.max_seq_len = opt.max_seq_len + 2 # include the <s> and </s>


    opt.train_locs = os.path.join(opt.data_root,opt.train_locs) 
    opt.train_src = os.path.join(opt.data_root,opt.train_src) 
    opt.train_tgt = os.path.join(opt.data_root,opt.train_tgt) 

    opt.valid_locs = os.path.join(opt.data_root,opt.valid_locs) 
    opt.valid_src = os.path.join(opt.data_root,opt.valid_src) 
    opt.valid_tgt = os.path.join(opt.data_root,opt.valid_tgt) 

    opt.test_locs = os.path.join(opt.data_root,opt.test_locs) 
    opt.test_src = os.path.join(opt.data_root,opt.test_src) 
    opt.test_tgt = os.path.join(opt.data_root,opt.test_tgt) 

    opt.save_data = os.path.join(opt.data_root,opt.save_data) 

    # Training set
    print('[Info] Building training insts')
    train_loc_insts = read_instances_from_file(opt.train_locs, -1, opt.keep_case,use_bos_eos=False)
    train_src_word_insts = read_instances_from_file(opt.train_src, opt.max_seq_len, opt.keep_case,use_bos_eos=False)
    train_tgt_word_insts = read_instances_from_file(opt.train_tgt, opt.max_tgt_len, opt.keep_case,use_bos_eos=False)
    assert len(train_src_word_insts) == len(train_tgt_word_insts) == len(train_loc_insts)

    # Validation set
    print('[Info] Building validation insts')
    valid_loc_insts = read_instances_from_file(opt.valid_locs, -1, opt.keep_case,use_bos_eos=False)
    valid_tgt_word_insts = read_instances_from_file(opt.valid_tgt, opt.max_tgt_len, opt.keep_case,use_bos_eos=False)
    valid_src_word_insts = read_instances_from_file(opt.valid_src, opt.max_seq_len, opt.keep_case,use_bos_eos=False)
    assert len(valid_src_word_insts) == len(valid_tgt_word_insts) == len(valid_loc_insts)

    print('[Info] Building testing insts')
    test_loc_insts = read_instances_from_file(opt.test_locs, -1, opt.keep_case,use_bos_eos=False)
    test_src_word_insts=read_instances_from_file(opt.test_src,opt.max_seq_len,opt.keep_case,use_bos_eos=False)
    test_tgt_word_insts=read_instances_from_file(opt.test_tgt,opt.max_tgt_len,opt.keep_case,use_bos_eos=False)
    assert len(test_src_word_insts) == len(test_tgt_word_insts) == len(test_loc_insts)

    # Build vocabulary
    if opt.vec_inputs:
        src_word2idx = None
    else:
        print('[Info] Build vocabulary for source.')
        src_word2idx = build_vocab_idx(train_src_word_insts, opt.min_word_count,use_bos_eos=False)

    print('[Info] Build vocabulary for target.')
    tgt_word2idx = build_vocab_idx(train_tgt_word_insts, opt.min_word_count,use_bos_eos=False)

    # word to index
    if opt.vec_inputs:
        train_src_insts =  [[float(y) for y in x] for x in train_src_word_insts]
        valid_src_insts = [[float(y) for y in x] for x in valid_src_word_insts] 
        test_src_insts = [[float(y) for y in x] for x in test_src_word_insts] 
    else:
        print('[Info] Convert source word instances into sequences of word index.')
        train_src_insts = convert_instance_to_idx_seq(train_src_word_insts, src_word2idx)
        valid_src_insts = convert_instance_to_idx_seq(valid_src_word_insts, src_word2idx)
        test_src_insts=convert_instance_to_idx_seq(test_src_word_insts, src_word2idx)

    print('[Info] Convert target word instances into sequences of word index.')
    train_tgt_insts = convert_instance_to_binary_vec(train_tgt_word_insts, tgt_word2idx)
    valid_tgt_insts = convert_instance_to_binary_vec(valid_tgt_word_insts, tgt_word2idx)
    test_tgt_insts = convert_instance_to_binary_vec(test_tgt_word_insts, tgt_word2idx)

    if opt.divide_split_into_chroms:

        train_chroms = divide_into_chroms(train_loc_insts,train_src_insts,train_tgt_insts)
        valid_chroms = divide_into_chroms(valid_loc_insts,valid_src_insts,valid_tgt_insts)
        test_chroms = divide_into_chroms(test_loc_insts,test_src_insts,test_tgt_insts)
        
        data = {
            # 'settings': opt,
            'dict': {
                'src': src_word2idx,
                'tgt': tgt_word2idx},
            'train': train_chroms,
            'valid': valid_chroms,
            'test': test_chroms
            }

    else:
        data = {
    
        # 'settings': opt,
        'dict': {
            'src': src_word2idx,
            'tgt': tgt_word2idx},
        'train': {
            'loc': train_loc_insts,
            'src': train_src_insts,
            'tgt': train_tgt_insts},
        'valid': {
            'loc': valid_loc_insts,
            'src': valid_src_insts,
            'tgt': valid_tgt_insts},
        'test': {
            'loc': test_loc_insts,
            'src': test_src_insts,
            'tgt': test_tgt_insts}
        
        }
    

    print('[Info] Dumping the processed data to torch file', opt.save_data)
    torch.save(data, opt.save_data)


if __name__ == '__main__':
    main()
