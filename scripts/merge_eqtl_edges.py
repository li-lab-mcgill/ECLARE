#%% set env variables
import os
import sys

# Check if environment variables are already set
eclare_root = os.environ.get('ECLARE_ROOT')
outpath = os.environ.get('OUTPATH')
datapath = os.environ.get('DATAPATH')

# Print status of environment variables
if all([eclare_root, outpath, datapath]):
    print(f"Environment variables already set:")
    print(f"ECLARE_ROOT: {eclare_root}")
    print(f"OUTPATH: {outpath}")
    print(f"DATAPATH: {datapath}")
else:
    print(f"Missing environment variables")

    config_path = '../config'
    sys.path.insert(0, config_path)

    from export_env_variables import export_env_variables
    export_env_variables(config_path)

#%%
import pandas as pd

eqtl_edges_dir = os.path.join(os.environ['DATAPATH'], 'brainSCOPE', 'eqtl_edges')
eqtl_edges_files = [f for f in os.listdir(eqtl_edges_dir) if f.endswith('.eqtl_edge.txt')]

eqtl_edges_dfs = []
for file in eqtl_edges_files:
    df = pd.read_csv(os.path.join(eqtl_edges_dir, file), sep='\t')
    celltype = file.split('.eqtl_edge.txt')[0]
    df['celltype'] = celltype
    eqtl_edges_dfs.append(df)

eqtl_edges_df = pd.concat(eqtl_edges_dfs, ignore_index=True)
assert (eqtl_edges_df['eqtl.Gene'] == eqtl_edges_df['GRN.TG']).all()

#eqtl_edges_df.to_csv(os.path.join(eqtl_edges_dir, 'eqtl_edges.txt'), sep='\t', index=False)

#%% see if have scQTL associated to gene hits

TFs = ['NR4A2', 'EGR1', 'SOX2']
TGs = ['EGR1', 'ABHD17B', 'PDE1C', 'TTYH2', 'OLIG2', 'DEDD2', 'APBB2', 'ZBED5']
hits = list(set(TFs + TGs))

eqtl_edges_df_hits = eqtl_edges_df[eqtl_edges_df['GRN.TG'].isin(hits)]

#%% load metaQTLs

metaqtl_dir = os.path.join(os.environ['DATAPATH'], 'brainSCOPE', 'metaQTLs.txt')
metaqtl_df = pd.read_csv(metaqtl_dir, header=None)

metaqtl_df_first_split = metaqtl_df[0].str.split('\t', expand=True)
first_rows = metaqtl_df_first_split.groupby(0).first()
print(first_rows)

dynamic_eqtl_genes = metaqtl_df_first_split[metaqtl_df_first_split[0]=='dynamic_eqtl'][1].str.split('_', expand=True)[0]
non_dynamic_eqtl_genes = metaqtl_df_first_split[metaqtl_df_first_split[0]!='dynamic_eqtl'][2]
eqtl_genes = pd.concat([dynamic_eqtl_genes, non_dynamic_eqtl_genes]).sort_index()

#%% see if have metaQTLs associated to gene hits

metaqtl_df_hits = metaqtl_df_first_split[eqtl_genes.isin(hits)]
metaqtl_genes_hits = eqtl_genes[eqtl_genes.isin(hits)].unique()
metaqtl_celltypes_hits = metaqtl_df_hits[1].unique()


# %%
