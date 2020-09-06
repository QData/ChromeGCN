''' Data Loader class for training iteration '''
import random
import numpy as np
import torch
import math
import data.Constants as Constants
from pdb import set_trace as stop
from utils import util_methods
from os import path


class DataLoader(object):
    ''' For data iteration '''

    def __init__(
            self, src_word2idx, tgt_word2idx,
            src_insts=None, loc_insts=None, tgt_insts=None,
            cuda=True, batch_size=64, shuffle=True,
            binary_relevance=False,drop_last=False):

        assert src_insts
        assert len(src_insts) >= batch_size

        if tgt_insts:
            assert len(src_insts) == len(tgt_insts)
        if loc_insts:
            assert len(src_insts) == len(loc_insts)
            self._loc_insts = loc_insts
        else:
            self._loc_insts = None

        self.cuda = cuda
        self._n_batch = int(np.ceil(len(src_insts) / batch_size))
        if drop_last:
            self._n_batch -= 1

        self._batch_size = batch_size
        self._src_insts = src_insts
        self._tgt_insts = tgt_insts

        if src_word2idx is not None:
            src_idx2word = {idx:word for word, idx in src_word2idx.items()}
            self._src_word2idx = src_word2idx
            self._src_idx2word = src_idx2word
            self.long_input = True
        else:
            self._src_word2idx = src_insts[0]
            self.long_input = False

        tgt_idx2word = {idx:word for word, idx in tgt_word2idx.items()}
        self._tgt_word2idx = tgt_word2idx
        self._tgt_idx2word = tgt_idx2word
        self._iter_count = 0
        self._need_shuffle = shuffle

        if self._need_shuffle:
            self.shuffle()

    @property
    def n_insts(self):
        ''' Property for dataset size '''
        return len(self._src_insts)

    @property
    def src_vocab_size(self):
        ''' Property for vocab size '''
        return len(self._src_word2idx)

    @property
    def tgt_vocab_size(self):
        ''' Property for vocab size '''
        return len(self._tgt_word2idx)

    @property
    def src_word2idx(self):
        ''' Property for word dictionary '''
        return self._src_word2idx

    @property
    def tgt_word2idx(self):
        ''' Property for word dictionary '''
        return self._tgt_word2idx

    @property
    def src_idx2word(self):
        ''' Property for index dictionary '''
        return self._src_idx2word

    @property
    def tgt_idx2word(self):
        ''' Property for index dictionary '''
        return self._tgt_idx2word

    def shuffle(self):
        ''' Shuffle data for a brand new start '''
        if self._tgt_insts:
            paired_insts = list(zip(self._loc_insts,self._src_insts,self._tgt_insts))
            random.shuffle(paired_insts)
            self._loc_insts, self._src_insts, self._tgt_insts = zip(*paired_insts)
        else:
            random.shuffle(self._src_insts)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        return self._n_batch

    def next(self):
        ''' Get the next batch '''

        def convert_string_to_mat(adj_string):
            dim = int(math.sqrt(len(adj_string)))
            output = torch.Tensor(adj_string).view(dim,dim)#.type(torch.uint8)

            if self.cuda:
                output = output.cuda()

            return(output)

        def construct_adj_mat(insts,encoder=False):

            inst_data_tensor = [convert_string_to_mat(inst) for inst in insts]

            return inst_data_tensor

        def pad_to_longest(insts,encoder=False):
            ''' Pad the instance to the max seq length in batch '''
            max_len = max(len(inst) for inst in insts)

            inst_data = np.array([
                inst + [Constants.PAD] * (max_len - len(inst))
                for inst in insts])

            inst_position = np.array([
                [pos_i+1 if w_i != Constants.PAD else 0 for pos_i, w_i in enumerate(inst)]
                for inst in inst_data])

            inst_data_tensor = torch.Tensor(inst_data)
            inst_position_tensor = torch.Tensor(inst_position)
            if self.cuda:
                inst_data_tensor = inst_data_tensor.cuda()
                inst_position_tensor = inst_position_tensor.cuda()
            return inst_data_tensor, inst_position_tensor

        if self._iter_count < self._n_batch:
            batch_idx = self._iter_count
            self._iter_count += 1
            start_idx = batch_idx * self._batch_size
            end_idx = (batch_idx + 1) * self._batch_size
            src_insts = self._src_insts[start_idx:end_idx]
            loc_insts = self._loc_insts[start_idx:end_idx]
            src_data, src_pos = pad_to_longest(src_insts,encoder=True)
            src_pos = src_pos.long()
            
            if self.long_input:
                src_data = src_data.long()

            if not self._tgt_insts:
                return src_data, src_pos
            else:
                tgt_insts = self._tgt_insts[start_idx:end_idx]
                tgt_data, tgt_pos = pad_to_longest(tgt_insts)
                tgt_data = tgt_data.long()
                tgt_pos = tgt_pos.long()
                return (src_data, src_pos), (loc_insts), tgt_data

        else:
            if self._need_shuffle:
                self.shuffle()

            self._iter_count = 0
            raise StopIteration()


def process_data(data,opt):
    if opt.summarize_data:
        util_methods.summarize_data(data)

     
    
    train_data = DataLoader(
        data['dict']['src'],
        data['dict']['tgt'],
        src_insts=data['train']['src'],
        loc_insts=data['train']['loc'],
        tgt_insts=data['train']['tgt'],
        batch_size=opt.batch_size,
        binary_relevance=True,
        cuda=opt.cuda,
        shuffle=opt.shuffle_train,
        drop_last=False)

    valid_data = DataLoader(
        data['dict']['src'],
        data['dict']['tgt'], 
        src_insts=data['valid']['src'],
        loc_insts=data['valid']['loc'],
        tgt_insts=data['valid']['tgt'],
        batch_size=opt.test_batch_size,
        binary_relevance=True,
        shuffle=False,
        cuda=opt.cuda)

    test_data = DataLoader(
        data['dict']['src'],
        data['dict']['tgt'], 
        src_insts=data['test']['src'],
        loc_insts=data['test']['loc'],
        tgt_insts=data['test']['tgt'],
        batch_size=opt.test_batch_size,
        binary_relevance=True,
        shuffle=False,
        cuda=opt.cuda)

    return train_data,valid_data,test_data