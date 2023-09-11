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


def preprocessing(adata: ad.AnnData,  hvgs, scale, scale_total, norm_total=False, log1p=False, pca=False, use_highly_variable=None, n_high_var=2000, flavor="seurat", hvg_list=None, n_comps=50, batch_key="batch", target_sum=1e4, scale_value=None):

    print("Started preprocessing...")

    temp_ad = adata.copy()

    if norm_total:
        sc.pp.normalize_total(temp_ad, target_sum=target_sum)
        
    if log1p:
        sc.pp.log1p(temp_ad)

    # HVG-ovi se racunaju po batch-u! subset ne mora da se uzima jer PCA po defaultu radi samo nad HVG-ovima ako su izracunati
    if hvgs:
        sc.pp.highly_variable_genes(temp_ad, n_top_genes=n_high_var, flavor=flavor, batch_key=batch_key, n_bins=20)
        temp_highly_variable = temp_ad.var.highly_variable.copy()

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
            temp_ad=ad.concat(adata_sep)

    if scale_total:
        sc.pp.scale(temp_ad,max_value=scale_value)

    # assign calculated HVGs (in case scale function has removed them)
    if hvgs:
        temp_ad.var['highly_variable'] = temp_highly_variable
        
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

    # elif(cluster_method=="kmeans"):
    #     X_pca = temp_ad.obsm[use_rep] 
    #     kmeans = KMeans(n_clusters=num_cluster, random_state=0).fit(X_pca) 
    #     temp_ad.obs['kmeans'] = kmeans.labels_.astype(str)
    #     temp_ad.obs['kmeans'] = temp_ad.obs['kmeans'].astype("category")   
        

    return temp_ad

def call_harmony(adata: ad.AnnData, label_name='Harmony_emb', batch_key='batch', n_comps=50):

    from harmony import harmonize

    sc.tl.pca(adata, n_comps=n_comps)

    print("Harmony...")     
    # only measure time for harmonize function
    t_init = time.process_time()
    adata.obsm[label_name] = harmonize(adata.obsm["X_pca"], adata.obs, batch_key=batch_key)
    t_fin = time.process_time()
    print(f'Harmonize function ran for: {t_fin-t_init} s')
    exe_time = t_fin-t_init
    return adata, exe_time

def call_scDML(adata: ad.AnnData, label_name='scDML_emb', verbose=False, initial_res=3.0, n_comps=50, hvgs=True, save_dir='./test_result/', batch_key='batch', ncluster=14, ncluster_list=[14], merge_rule="rule2"):

    import scDML
    from scDML import scDMLModel
    from scDML.utils import print_dataset_information

    print("scDML...")

    temp_ad = adata.copy()
    temp_ad.obs['BATCH'] = temp_ad.obs[batch_key]
    # get calculated HVGs
    if hvgs:
        hvg_list = list(temp_ad.var[temp_ad.var.highly_variable].index.values)
    else:
        hvg_list = list(temp_ad.var.index.values)

    scdml=scDMLModel(verbose=verbose, save_dir=save_dir)
    # since normalization, log1p, HVGs extraction and scaling were performed in the previous steps, we only let preprocess function to perform PCA
    adt = scdml.preprocess(temp_ad, cluster_method="louvain", resolution=initial_res, pca_dim=n_comps, hvg_list=hvg_list, normalize_samples=False, normalize_features=False, log_normalize=False, batch_key='BATCH')
    # only measure time for integrate function
    t_init = time.process_time()
    scdml.integrate(adt, batch_key='BATCH', ncluster_list=ncluster_list, expect_num_cluster=ncluster,merge_rule=merge_rule)
    t_fin = time.process_time()
    print(f'scDML integrate ran for: {t_fin-t_init} s')
    temp_ad.obsm[label_name] = adt.obsm["X_emb"]
    # del temp_ad.obsm['X_emb']

    exe_time = t_fin-t_init
    print("Adata after scDML: ", temp_ad)
    return temp_ad, exe_time

def call_liger(adata: ad.AnnData, label_name='Liger_emb', batch_key='batch', hvgs=True):

    import pyliger

    # parts of code were borrowed this tuturial: https://scib-metrics.readthedocs.io/en/stable/notebooks/lung_example.html#liger

    if hvgs:
        hvg = adata.var[adata.var.highly_variable].index.values
        bdata = adata[:, hvg].copy()
    else:
        bdata = adata.copy()

    # Pyliger normalizes by library size with a size factor of 1
    # So here we give it the count data
    bdata.X = csr_matrix(bdata.layers["counts"])
    # List of adata per batch
    batch_cats = np.unique(adata.obs[batch_key].values)
    adata_list = [bdata[bdata.obs[batch_key] == b].copy() for b in batch_cats]
    for i, ad in enumerate(adata_list):
        ad.uns["sample_name"] = batch_cats[i]
        # Hack to make sure each method uses the same genes
        ad.uns["var_gene_idx"] = np.arange(bdata.n_vars)


    liger_data = pyliger.create_liger(adata_list, remove_missing=False, make_sparse=False)
    # Hack to make sure each method uses the same genes
    liger_data.var_genes = bdata.var_names
    pyliger.normalize(liger_data)
    pyliger.scale_not_center(liger_data)
    # only measure time for optimize_ALS and quantile_norm
    t_init = time.process_time()
    pyliger.optimize_ALS(liger_data, k=30)
    pyliger.quantile_norm(liger_data)
    t_fin = time.process_time()

    exe_time = t_fin-t_init

    adata.obsm[label_name] = np.zeros((adata.shape[0], liger_data.adata_list[0].obsm["H_norm"].shape[1]))
    for i, b in enumerate(batch_cats):
        adata.obsm[label_name][adata.obs[batch_key] == b] = liger_data.adata_list[i].obsm["H_norm"]

    return adata, exe_time

def call_scVI(adata: ad.AnnData, label_name='scVI_emb', batch_key='batch', max_epochs=None, hvgs=True, raw_count_layer='counts'):
    # scVI models raw counts directly, so it is important that we provide it with a count matrix rather than a normalized expression matrix.
    import scvi
    from scvi.model import SCVI
    scvi.settings.seed = 0

    # code was borrowed from scVI official tutorial: https://docs.scvi-tools.org/en/stable/tutorials/notebooks/scrna/harmonization.html#dataset-preprocessing

    n_latent = 30
    n_hidden = 128
    n_layers = 2

    # copying to not return values added to adata during setup_anndata
    net_adata = adata.copy()
    if hvgs:
        hvg = net_adata.var[net_adata.var.highly_variable].index.values
        net_adata = adata[:, hvg].copy()

    # only measure time for training
    t_init = time.process_time()
    SCVI.setup_anndata(net_adata, layer=raw_count_layer, batch_key=batch_key)

    vae = SCVI(
        net_adata,
        gene_likelihood="nb",
        n_layers=n_layers,
        n_latent=n_latent,
        n_hidden=n_hidden,
    )
    train_kwargs = {"train_size": 1.0}
    if max_epochs is not None:
        train_kwargs["max_epochs"] = max_epochs
    vae.train(**train_kwargs)
    t_fin = time.process_time()
    adata.obsm[label_name] = vae.get_latent_representation()

    exe_time = t_fin-t_init
    return adata, exe_time


def save_results_to_dataframe(dataset, algorithm, hvgs, scaled_batch, scaled_total, graph_iLISI, graph_cLISI, ARI, NMI, ASW, exe_time, dataframe=None, path='/goofys/users/Aleksandra_S/benchmarking_datasets/results/', file_name_sufix = ''):
    
    file_path = path + 'result_table_new.csv'

    if dataframe is None:
        if os.path.exists(file_path):
            dataframe = pd.read_csv(file_path, index_col=0)

    if dataframe is None:
        dataframe = pd.DataFrame(columns=['dataset', 'algorithm', 'hvgs', 'scaled batch', 
        # 'scaled total', 
        'graph iLISI', 'graph cLISI', 'ARI', 'NMI', 'ASW', 'Execution time'])
    
    results_dict = {
        'dataset': dataset,
        'algorithm': algorithm,
        'hvgs': hvgs,
        'scaled batch': scaled_batch,
        # 'scaled total': scaled_total,
        'graph iLISI': graph_iLISI,
        'graph cLISI': graph_cLISI,
        'ARI': ARI,
        'NMI': NMI,
        'ASW': ASW,
        'Execution time': exe_time
    }
    
    dataframe = dataframe.append(results_dict, ignore_index=True)

    dataframe.to_csv(file_path)
    
    return dataframe


def main(path, params):

    use_rep = params.get('algo')+'_emb'
    graph_ilisi_k0 = params.get('graph_ilisi_k0')
    print("==========================================")
    # reading the data
    adata = ad.read_h5ad(path)
    print("Read anndata.")
    # adata.obs['batch'] = adata.obs.tech
    # preprocessing the data
    adata = preprocessing(adata, norm_total= params.get('norm_total'),  log1p= params.get('log1p'), hvgs=params.get('hvgs'), flavor=params.get('flavor'), scale=params.get('scale'), scale_total=params.get('scale_total'), batch_key=params.get('batch_key'), n_comps=params.get('n_comps'))
    print("Anndata after preprocessing: ", adata)
    # calling batch-removal method
    if params.get('algo').lower()=='harmony':
        adata, exe_time = call_harmony(adata, batch_key=params.get('batch_key'), n_comps=params.get('n_comps'), label_name=use_rep)
        print('Anndata after integration: ', adata)
    elif params.get('algo').lower()=='scdml':
        adata, exe_time = call_scDML(adata, batch_key=params.get('batch_key'), n_comps=params.get('n_comps'), label_name=use_rep, hvgs=params.get('hvgs'))
        print('Anndata after integration: ', adata)
    elif params.get('algo').lower()=='liger':
        # for liger we should not perform scaling (it will perform its own without zero-centering the data)
        adata, exe_time = call_liger(adata, batch_key=params.get('batch_key'), label_name=use_rep, hvgs=params.get('hvgs'))
        print('Anndata after integration: ', adata)
    elif params.get('algo').lower()=='scvi':
        # scVI models raw counts directly, so it is important that we provide it with a count matrix rather than a normalized expression matrix.
        # for scVI we should perform no scaling
        adata, exe_time = call_scVI(adata, batch_key=params.get('batch_key'), label_name=use_rep, raw_count_layer='counts', hvgs=params.get('hvgs'))
        print('Anndata after integration: ', adata)
    # performing clustering
    # resolutions = np.arange(params.get('resolutions_min'), params.get('resolutions_max'), params.get('resolutions_step'))
    # for r in resolutions:
    #     adata = clustering(adata, reso=r)
    #     key = "louvain"
    
    # adata = clustering(adata, cluster_method=params.get('clust_algo'), reso=params.get('reso'), use_rep=use_rep)
    # calculating metrics
    # ari=adjusted_rand_score(adata.obs[params.get('clust_algo')], adata.obs[params.get('cell_type_key')])
    # nmi=normalized_mutual_info_score(adata.obs[params.get('clust_algo')],adata.obs[params.get('cell_type_key')])
    clisi = scib.me.clisi_graph(adata, label_key=params.get('cell_type_key'), type_="embed", use_rep=use_rep)
    # print(f"Louvain resolution = {params.get('reso')}")
    # print(f"ARI = {ari}")
    # print(f"NMI = {nmi}")
    print(f"cLISI = {clisi}")

    if params.get('algo').lower()=='scdml':
        ilisi = scib.me.ilisi_graph(adata, batch_key='BATCH', type_="embed", use_rep=use_rep, k0=graph_ilisi_k0)
        asw = scib.me.silhouette_batch(adata, batch_key="BATCH", label_key=params.get('cell_type_key'), embed=use_rep, metric='euclidean')
    else:
        ilisi = scib.me.ilisi_graph(adata, batch_key=params.get('batch_key'), type_="embed", use_rep=use_rep, k0=graph_ilisi_k0)
        asw = scib.me.silhouette_batch(adata, batch_key=params.get('batch_key'), label_key=params.get('cell_type_key'), embed=use_rep, metric='euclidean')
    print(f"iLISI={ilisi}")
    print(f"ASW={asw}")

    print("Final andata:")
    print(adata) 
    print("Parameters: ")
    print(params)

    save_results_to_dataframe(dataset=params.get('dataset'), algorithm=params.get('algo'), hvgs=params.get('hvgs'), scaled_batch=params.get('scale'), scaled_total=params.get('scale_total'), graph_iLISI=ilisi, graph_cLISI=clisi, ARI=-1, NMI=-1, ASW=asw, exe_time=exe_time)
    # save anndata
    hvgs_name = "_hvgs" if params.get('hvgs') else ""
    scale_name = "_scale" if params.get('scale') else "" 
    adt_name = params.get('dataset') + "_" + params.get('algo') + hvgs_name + scale_name
    #adata.write_h5ad("/goofys/users/Aleksandra_S/benchmarking_datasets/results/" + adt_name + ".h5ad")
    adata.write_h5ad("results/" + adt_name + ".h5ad")


if __name__ == "__main__":
    params_list = [
    {
        'algo' : 'scVI',
        'dataset' : 'Lung_atlas_public',
        'hvgs' : False,
        'flavor' : 'cell_ranger',
        'scale' : False,
        'scale_total' : False,
        'norm_total' : False,
        'log1p' : False,
        'pca' : False,
        'resolutions_min' : 0.1,
        'resolutions_max' : 1.1,
        'resolutions_step' : 0.1,
        'reso' : 1,
        'clust_algo' : 'louvain',
        'batch_key' : 'batch',
        'cell_type_key' : 'cell_type',
        'n_comps' : 30,
        'graph_ilisi_k0' : 63,
    },
    # {
    #     'algo' : 'scVI',
    #     'dataset' : 'Lung_atlas_public',
    #     'hvgs' : False,
    #     'flavor' : 'cell_ranger',
    #     'scale' : True,
    #     'scale_total' : False,
    #     'norm_total' : False,
    #     'log1p' : False,
    #     'pca' : False,
    #     'resolutions_min' : 0.1,
    #     'resolutions_max' : 1.1,
    #     'resolutions_step' : 0.1,
    #     'reso' : 1,
    #     'clust_algo' : 'louvain',
    #     'batch_key' : 'batch',
    #     'cell_type_key' : 'cell_type',
    #     'n_comps' : 30,
    #     'graph_ilisi_k0' : 63,
    # },
    {
        'algo' : 'scVI',
        'dataset' : 'Lung_atlas_public',
        'hvgs' : True,
        'flavor' : 'cell_ranger',
        'scale' : False,
        'scale_total' : False,
        'norm_total' : False,
        'log1p' : False,
        'pca' : False,
        'resolutions_min' : 0.1,
        'resolutions_max' : 1.1,
        'resolutions_step' : 0.1,
        'reso' : 1,
        'clust_algo' : 'louvain',
        'batch_key' : 'batch',
        'cell_type_key' : 'cell_type',
        'n_comps' : 30,
        'graph_ilisi_k0' : 63,
    },
    # {
    #     'algo' : 'scVI',
    #     'dataset' : 'Lung_atlas_public',
    #     'hvgs' : True,
    #     'flavor' : 'cell_ranger',
    #     'scale' : True,
    #     'scale_total' : False,
    #     'norm_total' : False,
    #     'log1p' : False,
    #     'pca' : False,
    #     'resolutions_min' : 0.1,
    #     'resolutions_max' : 1.1,
    #     'resolutions_step' : 0.1,
    #     'reso' : 1,
    #     'clust_algo' : 'louvain',
    #     'batch_key' : 'batch',
    #     'cell_type_key' : 'cell_type',
    #     'n_comps' : 30,
    #     'graph_ilisi_k0' : 63,
    # },
    ]

    t_init = time.process_time()
    for params in params_list:
        main(
            '/goofys/users/Aleksandra_S/benchmarking_datasets/Lung_atlas_public.h5ad', params
        )
    t_fin = time.process_time()
    print(f'Total time = {t_fin-t_init} s')
