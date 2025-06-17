import os
import pandas as pd
import numpy as np
import networkx as nx
import scglue
import anndata
from tqdm import tqdm

from pyjaspar import jaspardb
import pyranges as pr
from Bio.motifs.jaspar import calculate_pseudocounts

from src.eclare.data_utils import get_unified_grns


# Get mean GRN from brainSCOPE & scglue preprocessing

grn_path = os.path.join(os.environ['DATAPATH'], 'brainSCOPE', 'GRNs')
mean_grn_df = get_unified_grns(grn_path)

## get gene annotation and position
#scglue.data.get_gene_annotation(
#    mdd_rna, gtf=os.path.join(os.environ['DATAPATH'], 'gencode.v48.annotation.gtf.gz'),#
#    gtf_by="gene_name"
#)

## get peak position
split = mean_grn_df['enhancer'].str.split(r"[:-]")
mean_grn_df["chrom"] = split.map(lambda x: x[0])
mean_grn_df["chromStart"] = split.map(lambda x: x[1]).astype(int)
mean_grn_df["chromEnd"] = split.map(lambda x: x[2]).astype(int)

## extract gene and peak positions
gene_ad = anndata.AnnData(var=pd.DataFrame(index=mean_grn_df['TG'].unique()))
scglue.data.get_gene_annotation(gene_ad, gtf=os.path.join(os.environ['DATAPATH'], 'gencode.v48.annotation.gtf.gz'), gtf_by='gene_name')
gene_ad = gene_ad[:,gene_ad.var['name'].notna()]

genes = scglue.genomics.Bed(gene_ad.var)
tss = genes.strand_specific_start_site()
promoters = tss.expand(2000, 0)

peaks = scglue.genomics.Bed(mean_grn_df.assign(name=mean_grn_df['enhancer'])).drop_duplicates()

## distance graph
window_size = 1e6
dist_graph = scglue.genomics.window_graph(
    promoters, peaks, window_size,
    attr_fn=lambda l, r, d: {
        "dist": abs(d),
        "weight": scglue.genomics.dist_power_decay(abs(d)),
        "type": "dist"
    }
)
dist_graph = nx.DiGraph(dist_graph)
dist_graph.number_of_edges()

def graph_to_df(G):
    """
    Convert a NetworkX DiGraph G into a pandas DataFrame
    with columns: source, target, weight, type (and any other attrs).
    """
    rows = []
    for u, v, attrs in G.edges(data=True):
        row = {
            "source": u,
            "target": v,
            **attrs
        }
        rows.append(row)
    return pd.DataFrame(rows)

# e.g. for the distance graph
df_dist = graph_to_df(dist_graph)
print(df_dist.head())

## merge distance graph with mean_grn_df
mean_grn_df = mean_grn_df.merge(df_dist, left_on=['TG', 'enhancer'], right_on=['source', 'target'], how='left')

## get JASPAR TFs db
jaspar_db_2020 = jaspardb(release='JASPAR2020')
jaspar_db_2024 = jaspardb(release='JASPAR2024')

tfs = list(mean_grn_df['TF'].unique())
#motifs = {tf:jaspar_db.fetch_motifs(tf_name=tf, tax_group='vertebrates') for tf in tfs} # ~25 seconds
#motifs = jaspar_db_2018.fetch_motifs(tf_name=tfs, tax_group='vertebrates')

mean_grn_df['motif_score'] = 0
tf_enhancer_grns = list(mean_grn_df.groupby('TF')['enhancer'].unique().items())

for tf, enhancers in tqdm(tf_enhancer_grns, total=len(tf_enhancer_grns)):

    try:
        motifs = jaspar_db_2020.fetch_motifs(tf_name=tf, tax_group='vertebrates')
        motif = motifs[0]
    except:
        try:
            motifs = jaspar_db_2024.fetch_motifs_by_name(tf)
            motif = motifs[0]
        except:
            print(f"Could not find motif for TF {tf} in either JASPAR2020 or JASPAR2024")
            continue

    motif.pseudocounts = calculate_pseudocounts(motif)  # also have motif.pseudo_counts, not sure what the difference is...

    enhancers_df = pd.DataFrame(enhancers)[0].str.split(':|-', expand=True).rename(columns={0: 'Chromosome', 1: 'Start', 2: 'End'})
    gr = pr.from_dict(enhancers_df.to_dict()) # can extend with pr.extend(k)
    seqs = pr.get_sequence(gr, os.path.join(os.environ['DATAPATH'], 'hg38.fa')) # ~53 seconds

    motif_scores = {enhancer: motif.pssm.calculate(seq.upper()).max() for enhancer, seq in zip(enhancers, seqs)} # ~44 seconds
    mean_grn_df.loc[mean_grn_df['TF'] == tf, 'motif_score'] = mean_grn_df.loc[mean_grn_df['TF'] == tf, 'enhancer'].map(motif_scores)

## remove TFs with no motif score
print(f"Removing {mean_grn_df[mean_grn_df['motif_score'] == 0]['TF'].nunique()} TFs out of {mean_grn_df['TF'].nunique()} with no motif score")
mean_grn_df = mean_grn_df[mean_grn_df['motif_score'] > 0]
mean_grn_df.reset_index(drop=True, inplace=True)  # need to reset indices to enable alignment with normed values

## normalize motif score by target gene TG
#motif_score_norm = mean_grn_df.groupby('TG')['motif_score'].apply(lambda x: (x - x.min()) / (x.max() - x.min())) # guaranteed to have one zero-score per TG
temperature = mean_grn_df['motif_score'].var()
softmax_temp_func = lambda x, temperature: np.exp(x / temperature)# / np.exp(x / temperature).sum()
motif_score_norm = mean_grn_df.groupby('TG')['motif_score'].apply(lambda x: softmax_temp_func(x, temperature))
motif_score_norm = motif_score_norm.fillna(1)  # NaN motif scores because only one motif score for some TGs

motif_score_norm = motif_score_norm.reset_index(level=0, drop=False).rename(columns={'motif_score': 'motif_score_norm'})
assert (motif_score_norm.index.sort_values() == np.arange(len(motif_score_norm))).all()

## merge motif scores with mean_grn_df
mean_grn_df = mean_grn_df.merge(motif_score_norm, left_index=True, right_index=True, how='left', suffixes=('', '_motifs'))

mean_grn_df.to_pickle(os.path.join(os.environ['OUTPATH'], 'mean_grn_df.pkl'))