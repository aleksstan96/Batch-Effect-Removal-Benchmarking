import anndata as ad
import scanpy as sc 
import seaborn as sns
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
import torch
from scipy import stats
import scib
# import scvi
from sklearn.metrics.cluster import adjusted_rand_score,normalized_mutual_info_score
from scipy.sparse import csr_matrix

def clustering(adata: ad.AnnData, cluster_method="louvain", use_rep='X_pca', reso=1.0, num_cluster=10): 

    temp_ad = adata.copy()
    # del temp_ad.obsp
    # del temp_ad.obs['louvain']
    # del temp_ad.uns['neighbors']

    if(cluster_method=="leiden"):
        # if 'neighbors' not in temp_ad.uns:
        sc.pp.neighbors(temp_ad, use_rep=use_rep,random_state=0)
        sc.tl.leiden(temp_ad,resolution=reso,key_added=cluster_method)

    elif(cluster_method=="louvain"):
        # if 'neighbors' not in temp_ad.uns:
        sc.pp.neighbors(temp_ad, use_rep=use_rep,random_state=0)
        sc.tl.louvain(temp_ad,resolution=reso,key_added=cluster_method)


def main(path, params):
    
    use_rep= params.get('use_rep')
    adata = ad.read_h5ad(path)
    print("Read anndata from: ", path)
    print(adata)
    # performing clustering
    resolutions = np.arange(0.1, 1, 0.1)
    max_nmi = 0
    max_ari = 0
    for r in resolutions:
        key = "louvain"
        adata = clustering(adata, reso=r, cluster_method=key, use_rep=use_rep)
        # calculating metrics
        ari=adjusted_rand_score(adata.obs[params.get('clust_algo')], adata.obs[params.get('cell_type_key')])
        nmi=normalized_mutual_info_score(adata.obs[params.get('clust_algo')],adata.obs[params.get('cell_type_key')])
        if ari>max_ari:
            max_ari=ari
        if nmi>max_nmi:
            max_nmi=nmi
    # print(f"Louvain resolution = {params.get('reso')}")
    print(f"ARI = {max_ari}")
    print(f"NMI = {max_nmi}")

if __name__ == "__main__":

    params = {
        "use_rep" : "scDML_emb",
        'cell_type_key' : 'cell_type',
    }

    t_init = time.process_time()
    main(
        'results/Lung_atlas_public_scDML.h5ad', params
    )
    t_fin = time.process_time()
    print(f'Total time = {t_fin-t_init} s')