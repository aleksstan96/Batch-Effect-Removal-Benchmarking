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
from harmony import harmonize
import scib
# import scvi
import scDML
from scDML import scDMLModel
from scDML.utils import print_dataset_information
from sklearn.metrics.cluster import adjusted_rand_score,normalized_mutual_info_score


def preprocessing(adata: ad.AnnData, scale=True, norm_total=True, log1p=True, pca=True, use_highly_variable=None, hvgs = True, n_high_var=2000, flavor="cell_ranger", hvg_list=None, n_comps=100, batch_key="batch", target_sum=1e4, scale_value=None):

    print("Started preprocessing...")

    temp_ad = adata.copy()

    if norm_total:
        sc.pp.normalize_total(temp_ad, target_sum=target_sum)
        
    if log1p:
        sc.pp.log1p(temp_ad)

    # HVG-ovi se racunaju po batch-u! subset ne mora da se uzima jer PCA po defaultu radi samo nad HVG-ovima ako su izracunati
    if hvgs:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_high_var, flavor=flavor, batch_key=batch_key, n_bins=20)

    # scale se radi po batch-u!
    if scale: 
        if (batch_key not in temp_ad.obs.columns) or (len(temp_ad.obs[batch_key].unique())==1): #ako nema kolonu za batch, single batch, ili ima kolonu ali je samo jedan batch
            sc.pp.scale(temp_ad,max_value=scale_value)
            temp_ad.obs["batch"] = 1
        else:
            adata_sep=[]
            for batch in np.unique(temp_ad.obs[batch_key]):
                sep_batch=temp_ad[temp_ad.obs[batch_key]==batch].copy()
                sc.pp.scale(sep_batch,max_value=scale_value)
                adata_sep.append(sep_batch)
            temp_ad=sc.AnnData.concatenate(*adata_sep)
        
    if pca:
        # by default uses calculated HVGs
        sc.tl.pca(temp_ad,n_comps=n_comps, use_highly_variable=use_highly_variable)

    return temp_ad

def clustering(adata: ad.AnnData, cluster_method="louvain", use_rep='X_pca', reso=1.0, num_cluster=10): 

    temp_ad = adata.copy()

    if(cluster_method=="leiden"):
        if 'neighbors' not in temp_ad.uns:
            sc.pp.neighbors(temp_ad, use_rep=use_rep,random_state=0)
        sc.tl.leiden(temp_ad,resolution=reso,key_added=cluster_method)

    elif(cluster_method=="louvain"):
        if 'neighbors' not in temp_ad.uns:
            sc.pp.neighbors(temp_ad, use_rep=use_rep,random_state=0)
        sc.tl.louvain(temp_ad,resolution=reso,key_added=cluster_method)

    elif(cluster_method=="kmeans"):
        X_pca = temp_ad.obsm[use_rep] 
        kmeans = KMeans(n_clusters=num_cluster, random_state=0).fit(X_pca) 
        temp_ad.obs['kmeans'] = kmeans.labels_.astype(str)
        temp_ad.obs['kmeans'] = temp_ad.obs['kmeans'].astype("category")   
        

    return temp_ad


def call_harmony(adata: ad.AnnData, label_name='Harmony_emb', batch_key='batch'):
    print("Harmony...")
    t_init = time.process_time()
    adata.obsm[label_name] = harmonize(adata.obsm["X_pca"], adata.obs, batch_key=batch_key)
    t_fin = time.process_time()
    print(f'Harmonize function ran for: {t_fin-t_init} s')

    return adata

def call_scDML(adata: ad.AnnData, label_name='scDML_emb', verbose=False, save_dir='./test_result/', batch_key='batch', ncluster=14, ncluster_list=[14], merge_rule="rule2"):
    print("scDML...")

    temp_ad = adata.copy()
    t_init = time.process_time()

    scdml=scDMLModel(verbose=verbose, save_dir=save_dir)
    scdml.integrate(temp_ad, batch_key=batch_key, ncluster_list=ncluster_list, expect_num_cluster=ncluster,merge_rule=merge_rule)
    t_fin = time.process_time()
    print(f'scDML integrate ran for: {t_fin-t_init} s')
    temp_ad.obsm[label_name] = temp_ad.obsm["X_emb"]
    del temp_ad.obsm['X_emb']

    return temp_ad


def main(path):

    print("==========================================")

    adata = ad.read_h5ad(path)
    print("Read anndata.")
    adata.obs['batch'] = adata.obs.tech
    adata  = preprocessing(adata, hvgs=False)
    adata = call_harmony(adata)

    resolutions = np.arange(0.1, 1.1, 0.1)
    for r in resolutions:
        adata = clustering(adata, reso=r)
        key = "louvain"

        ari=adjusted_rand_score(adata.obs[key], adata.obs["celltype"])
        nmi=normalized_mutual_info_score(adata.obs[key],adata.obs["celltype"])
        clisi = scib.me.clisi_graph(adata, label_key="celltype", type_="embed", use_rep="Harmony_emb")
        print(f"Louvain resolution={r}, ARI={ari}")
        print(f"Louvain resolution={r}, NMI={nmi}")
        print(f"Louvain resolution={r}, cLISI={clisi}")

    ilisi = scib.me.ilisi_graph(adata, batch_key="batch", type_="embed", use_rep="Harmony_emb")
    print(f"iLISI={ilisi}")

    print("Final andata:")
    print(adata)

if __name__ == "__main__":
    t_init = time.process_time()
    main(
        '/goofys/users/Aleksandra_S/benchmarking_datasets/human_pancreas_norm_complexBatch.h5ad',
    )
    t_fin = time.process_time()
    print(f'Total time = {t_fin-t_init} s')
