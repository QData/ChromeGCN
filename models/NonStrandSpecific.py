"""
Computes the forward and reverse complement passes of an input sequence.

Taken from the Selene repository (Chen et. al. 2018)
https://github.com/FunctionLab/selene/
https://www.biorxiv.org/content/10.1101/438291v3.full

"""
import torch
from torch.nn.modules import Module
from pdb import set_trace as stop

def _flip(x, dim):
    """
    Reverses the elements in a given dimension `dim` of the Tensor.

    source: https://github.com/pytorch/pytorch/issues/229
    """
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.contiguous()
    x = x.view(-1, *xsize[dim:])
    x = x.view(
        x.size(0), x.size(1), -1)[:, getattr(
            torch.arange(x.size(1)-1, -1, -1),
            ('cpu','cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)

def _complement(x,src_dict):
    """
    A->T
    T->A
    C->G
    G->C

    """
    comp = x.clone()

    comp[x==src_dict['a']] = src_dict['t']
    comp[x==src_dict['t']] = src_dict['a']
    comp[x==src_dict['c']] = src_dict['g']
    comp[x==src_dict['g']] = src_dict['c']

    return comp


class GraphNonStrandSpecific(Module):
    """
    A torch.nn.Module that wraps a user-specified model architecture if the
    architecture does not need to account for sequence strand-specificity.

    Parameters
    ----------
    model : torch.nn.Module
        The user-specified model architecture.
    mode : {'mean', 'max'}, optional
        Default is 'mean'. NonStrandSpecific will pass the input and the
        reverse-complement of the input into `model`. The mode specifies
        whether we should output the mean or max of the predictions as
        the non-strand specific prediction.

    Attributes
    ----------
    model : torch.nn.Module
        The user-specified model architecture.
    mode : {'mean', 'max'}
        How to handle outputting a non-strand specific prediction.

    """

    def __init__(self, model, mode="mean"):
        super(GraphNonStrandSpecific, self).__init__()

        self.model = model

        if mode != "mean" and mode != "max":
            raise ValueError("Mode should be one of 'mean' or 'max' but was"
                             "{0}.".format(mode))
        self.mode = mode

    def forward(self, src, src_dict=None):
        
        # reverse_input = _flip(_flip(src, 1), 2) # for one hot
        reverse_comp = _flip(src, 1)
        reverse_comp = _complement(reverse_comp,src_dict)
    
        x_out,y_out,attn = self.model.forward(src)

        x_out_rev,y_out_rev,attn_rev = self.model.forward(reverse_comp)
        if self.mode == "mean":
            if attn is not None:
                return x_out,x_out_rev, (y_out + y_out_rev) / 2, attn, attn_rev
            else:
                return x_out,x_out_rev, (y_out + y_out_rev)  / 2, None, None
            # return (x_out + x_out_rev) / 2, (y_out + y_out_rev)  / 2
        else:
            return torch.max(output, output_from_rev), (enc_output_f,enc_output_r)

