#!/usr/bin/env python3
"""Removing batch effects from ST-seq experiment data
"""

__author__ = "Milos Radonjic"
__copyright__ = ""
__credits__ = ["Milos Radonjic"]
__license__ = ""
__version__ = "1.0.1"
__maintainer__ = "Milos Radonjic"
__email__ = "milosradonjic1@genomics.cn"
__status__ = "Production"

import math
import numpy as np
import pandas as pd
import scipy as sp
import scanpy as sc
import anndata as ad
from typing import Tuple
from multipledispatch import dispatch
import scipy.stats as stats
from scipy.stats import kurtosis
import statistics

def hardcoded_w(index : int):
    EdgeWeightArray = np.array([0.02, 0.2, 0.8, 1.0])
    if index < EdgeWeightArray.size:
        tmp = EdgeWeightArray[index]
    else:
        tmp = 1.0
    return tmp

def edge_weight_from_expression(_expression, _T=0.7):
    return 2.0/(1.0+math.exp((0.9999986-_expression)/_T))-1.0

def edge_weight_from_node_weight_and_expression(_node_weight, _node_expression):
    return edge_weight_from_expression(_node_expression)**(math.exp(1.0 - _node_weight))

def edge_weight_from_ew_per_group(wl):
    if wl.size > 1:
        imaxw = np.argmax(wl)
        maxw = wl[imaxw]
        if maxw<=0.99999:
            winc = (1.0-maxw)/(wl.size-1)
            return np.sum(np.delete(wl, imaxw)*winc)
        else:
            return 1.0
    else:
        return wl[0]

def edge_weight_from_node_weights_and_expressions_lists(_node_weights, _node_expressions):
    w = np.vectorize(edge_weight_from_node_weight_and_expression)
    wn = w(_node_weights, _node_expressions)
    return  edge_weight_from_ew_per_group(wn)

def edge_weight_from_total_expression(_node_expr, _tot_expr):
    return (_node_expr)/(_tot_expr)
    # return math.log1p(_node_expr)/math.log1p(_tot_expr)

def only_box_slice_ad(_adata : ad.AnnData, _x : int, _y : int, _r : int):
    """
    Slices specific AnnData object to [x-r, x+r]and[y-r, y+r].
    Keeps repeated genes in the list. That information is used for creating graph!
    """
    spot_slice = \
        (_adata.obs['x'] >= _x-_r) & \
        (_adata.obs['x'] <= _x+_r) & \
        (_adata.obs['y'] >= _y-_r) & \
        (_adata.obs['y'] <= _y+_r)
    _adata = _adata[spot_slice]
    return _adata[:, np.unique(_adata.X.indices)]

def band_slice_ad(_adata : ad.AnnData, _x : int, _y : int, _r : Tuple[int, int]):
    """
    Slices specific AnnData object to
        bin(2*r_max+1) - bin(2*r_min+1).
    Keeps repeated genes in the list. That information is used for creating graph!
    """
    spot_slice =\
        ( (_adata.obs['x'] >= _x-_r[1]) & (_adata.obs['x'] <= _x+_r[1]) & \
          (_adata.obs['y'] >= _y-_r[1]) & (_adata.obs['y'] <= _y+_r[1]) & \
        ( (_adata.obs['x'] <  _x-_r[0]) | (_adata.obs['x'] >  _x+_r[0]) | \
          (_adata.obs['y'] <  _y-_r[0]) | (_adata.obs['y'] >  _y+_r[0]) \
        )\
        )
    _adata = _adata[spot_slice]
    # print(f'Observed segment size: perimeter of bin{2*_r[0]+2} to bin{2*_r[1]+1}, (both edges included).\tempty spots ratio = {1.0-_adata.n_obs/(4*(_r[1]-_r[0])*(_r[0]+_r[1]+1))}')
    return _adata[:, np.unique(_adata.X.indices)]

@dispatch(_adata=ad.AnnData, _x=int, _y=int, _r=int)
def box_slice_ad(_adata : ad.AnnData, _x : int, _y : int, _r : int):
    return only_box_slice_ad(_adata=_adata, _x=_x, _y=_y, _r=_r)

@dispatch(_adata=ad.AnnData, _x=int, _y=int, _r=int, original_indexes=bool)
def box_slice_ad(_adata : ad.AnnData, _x : int, _y : int, _r : int, original_indexes : bool):
    tmp_ad = only_box_slice_ad(_adata=_adata, _x=_x, _y=_y, _r=_r)   
    return tmp_ad, np.vectorize(_adata.var.index.get_loc)(tmp_ad.var.index)

def flatten_list_of_lists(_l : list):
    L2 = []
    for l in _l:
        L2.extend(l)
    return L2

def get_sparse_arrays_from_LGraph(_LG ):
    le = list(zip(*_LG.G.edges('Template', 'weight') ))
    le_indexes = np.vectorize(_LG._ad.var.index.get_loc)(le[1])
    datae = np.array(le[2])
    return sp.sparse.csr_array((datae, (np.zeros(le_indexes.size), le_indexes)), shape=(1, _LG._ad.n_vars))

def filter_genes_by_expression(adata: ad.AnnData, min_total_expression=20, avg_expression_threshold=1.2, inplace=False):
    # Check if the qc metrics attributes are present
    if (inplace):
        _adata = adata
    else:
        _adata = adata.copy()
    if 'n_genes_by_counts' in _adata.obs.keys() and 'n_cells_by_counts' in _adata.var.keys():
        print("QC metrics calculation have been performed.")
    else:
        print("QC metrics calculation have not been performed, performing Scanpy calculate_qc_metrics function.")
        sc.pp.calculate_qc_metrics(_adata, inplace=True)
    # Remove all mitochondiral genes
    mito_genes_list = [name for name in _adata.var_names if name.startswith('mt-')]
    print(f"Removing mitochondrial genes, total {len(mito_genes_list)} of them")
    _adata = _adata [:, ~_adata.var_names.isin(mito_genes_list)]
    # Remove genes with total expression of 1
    total_counts_less_than_1_mask = _adata.var.total_counts<2.0
    genes_total_counts_1_indices = _adata.var[total_counts_less_than_1_mask].index
    print(f"Removing genes with total expression of 1, total {len(genes_total_counts_1_indices)} of them")
    _adata = _adata [:, ~_adata.var_names.isin(genes_total_counts_1_indices)]
    # Remove genes with average expression per spot < 1.2 (only spots that contain the selected gene are considered)
    total_counts_less_than_x_mask = _adata.var.total_counts<min_total_expression
    total_counts_less_than_x_indices  = _adata.var[total_counts_less_than_x_mask].total_counts.index
    genes_to_remove = [gene for gene in total_counts_less_than_x_indices if _adata[:, gene].var.total_counts[0]/_adata[:, gene].var.n_cells_by_counts[0]<=1.2]
    print(f"Removing genes with average expression per spot < {avg_expression_threshold}, total {len(genes_to_remove)} of them")
    _adata = _adata [:, ~_adata.var_names.isin(genes_to_remove)]
    return _adata


def filter_spots_by_gene_expression(adata: ad.AnnData, inplace=False):
   if (inplace):
        _adata = adata
   else:
        _adata = adata.copy()
   if 'n_genes_by_counts' in _adata.obs.keys() and 'n_cells_by_counts' in _adata.var.keys():
        print("QC metrics calculation have been performed.")
   else:
        print("QC metrics calculation have not been performed, performing Scanpy calculate_qc_metrics function.")
        sc.pp.calculate_qc_metrics(_adata, inplace=True)
   total_cnt_condition = _adata.obs['total_counts']
   cut = max(_adata.obs['total_counts']) - 1
   total = _adata.obs.total_counts.index
   #Remove all spots where gene expression is too high (kurtosis tail bigger than variance)
   while kurtosis(total_cnt_condition) > statistics.variance(total_cnt_condition):
       total_cnt_mask = _adata.obs.total_counts<cut
       total_cnt_condition = _adata[total_cnt_mask].obs.total_counts
       cut = cut/2
   _adata = _adata[_adata.obs['total_counts']<cut]
   total_cnt_new = _adata.obs.total_counts.index
   print(f"Removing spots with total expression kurtosis bigger than variance, total {len(total)-len(total_cnt_new)} of them")
   return _adata

def filter_spots_by_gene_expression_and_gene_count(df: pd.DataFrame, inplace=False):
    if (inplace):
        _df = df
    else:
        _df = df.copy() 
    df1 = _df.drop(['ExonCount', 'geneID'], axis=1)\
        .groupby(['y', 'x'])["MIDCount"]\
        .agg(['sum','size'])\
        .reset_index()
    # by gene expression
    cut = max(df1['sum']) - 1
    total_cnt_condition = df1['sum']
    while kurtosis(total_cnt_condition) > 2*(np.var(total_cnt_condition)):
        total_cnt_mask = df1['sum']<cut
        total_cnt_condition = df1[total_cnt_mask]['sum']
        cut = cut/2
    print(f"Removing spots with total expression less than {cut}, total {len(df1['sum'])-len(df1[df1['sum']<cut])} of them")
    df1 = df1[df1['sum']<cut].copy()
    # by gene count
    cut = max(df1['size']) - 1
    total_cnt_condition = df1['size']
    while kurtosis(total_cnt_condition) > 2*(np.var(total_cnt_condition)):
        total_cnt_mask = df1['size']<cut
        total_cnt_condition = df1[total_cnt_mask]['size']
        cut = cut/2
    print(f"Removing spots with gene count less than {cut}, total {len(df1['size'])-len(df1[df1['size']<cut])} of them")
    df1 = df1[df1['size']<cut].copy()
    _df = _df.merge(df1, how = 'inner', on = ['y', 'x'])
    return _df


def filter_unique_genes_by_spot(adata: ad.AnnData, inplace=False):
   if (inplace):
        _adata = adata
   else:
        _adata = adata.copy()
   if 'n_genes_by_counts' in _adata.obs.keys() and 'n_cells_by_counts' in _adata.var.keys():
        print("QC metrics calculation have been performed.")
   else:
        print("QC metrics calculation have not been performed, performing Scanpy calculate_qc_metrics function.")
        sc.pp.calculate_qc_metrics(_adata, inplace=True)
   genes_by_count_condition = _adata.obs['n_genes_by_counts']
   num = max(_adata.obs['n_genes_by_counts']) - 1
   total_genes = _adata.obs.n_genes_by_counts.index
    #Remove spots with more than a number of genes
   while kurtosis(genes_by_count_condition) > statistics.variance(genes_by_count_condition):
       gene_cnt_mask = _adata.obs.total_counts<num
       genes_by_count_condition = _adata[gene_cnt_mask].obs.total_counts
       num = num/2
   _adata = _adata[_adata.obs['total_counts']<num]
   gene_cnt_new = _adata.obs.total_counts.index
   print(f"Removing spots with with too many genes, total {len(total_genes)-len(gene_cnt_new)} of them")
   return _adata


def delete_module(modname: str, paranoid=None):
    from sys import modules
    try:
        thismod = modules[modname]
    except KeyError:
        raise ValueError(modname)
    these_symbols = dir(thismod)
    if paranoid:
        try:
            paranoid[:]  # sequence support
        except:
            raise ValueError('must supply a finite list for paranoid')
        else:
            these_symbols = paranoid[:]
    del modules[modname]
    for mod in modules.values():
        try:
            delattr(mod, modname)
        except AttributeError:
            pass
        if paranoid:
            for symbol in these_symbols:
                if symbol[:2] == '__':  # ignore special symbols
                    continue
                try:
                    delattr(mod, symbol)
                except AttributeError:
                    pass