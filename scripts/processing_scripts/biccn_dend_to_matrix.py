from eclare import set_env_variables
set_env_variables(config_path='../../config')

import os
import json
from io import StringIO
from Bio import Phylo
import numpy as np
import seaborn as sns
import pandas as pd
from itertools import combinations
import matplotlib.pyplot as plt

def quote_label(s):
    s = s.replace("'", r"\'")
    return f"'{s}'"

def json_to_newick(d):
    """
    Recursively build a Newick string from your nested JSON,
    guarding for children that are stored under 'leaf_attributes',
    and quoting labels to preserve spaces.
    """
    # Leaf node case
    if 'leaf_attributes' in d:
        leaf = d['leaf_attributes'][0]
        name = leaf['original_label']
        name = quote_label(name)
        return f"{name}:{leaf['height']:.4f}"

    # Internal node
    attrs    = d['node_attributes'][0]
    raw_name = attrs.get('label', '') or attrs.get('original_label', '')
    name     = quote_label(raw_name)
    children = d.get('children', [])

    subs      = []
    child_hts = []
    for c in children:
        subs.append(json_to_newick(c))
        if 'node_attributes' in c:
            child_hts.append(c['node_attributes'][0]['height'])
        else:
            child_hts.append(c['leaf_attributes'][0]['height'])

    # Calculate branch length
    length = attrs['height'] - max(child_hts)
    return f"({','.join(subs)}){name}:{length:.4f}"


def find_largest_distance(tree):
    all_clades = list(tree.find_clades())
    maxd = 0.0
    pair = (None, None)

    for a, b in combinations(all_clades, 2):
        # skip any unnamed clades if necessary
        if a.name is None or b.name is None: 
            continue
        d = tree.distance(a, b)
        if d > maxd:
            maxd = d
            pair = (a.name, b.name)

    print("Tree diameter (any node):", maxd)
    print("Between nodes:", pair)

    return maxd, pair

def get_celltype_similarity_matrix(filename):

    dend_path = os.path.join(os.environ['DATAPATH'], filename)

    if filename == 'dend_multiple_cortical_human.json':
        celltypes = ['Astrocyte', 'Endo L2-5 CLDN5', 'Excitatory', 'Inhibitory', 'Micro L1-6 C1QC', 'OPC L1-6 MYT1', 'Oligodendrocyte', 'Peri L1-6 MUSTN1']

    elif filename == 'dend_mtg_human.json':
        celltypes = ['Astrocyte', 'Endothelial', 'Neuronal: Glutamatergic', 'Neuronal: GABAergic', 'Microglia-PVM', 'OPC', 'Oligodendrocyte']

    elif filename == 'dend_ctx_hpc_mouse.json':
        celltypes = ['CTX-HPF 376-378', 'CTX-HPF 379', 'CTX-HPF 124-360', 'CTX-HPF 002-123', 'CTX-HPF 386-388', 'OPC', 'CTX-HPF 365-375']

    tree_json = json.load(open(dend_path))
    newick    = json_to_newick(tree_json) + ';'
    tree      = Phylo.read(StringIO(newick), "newick")

    names = []
    for clade in tree.find_clades():
        names.append(clade.name)

    leaf_labels = [term.name for term in tree.get_terminals()]
    names += leaf_labels
    names = np.array(names)
    names = names[names != None]
    names_sorted = np.sort(names)

    for celltype in celltypes:
        assert np.isin(celltype, names_sorted), f"{celltype} not found in 'names_sorted'"

    # Create a list of tuples for each pairwise comparison of celltypes
    pairwise_comparisons_df = pd.DataFrame(index=celltypes, columns=celltypes)

    for celltype1, celltype2 in combinations(celltypes, 2):
        distance = tree.distance(celltype1, celltype2)
        pairwise_comparisons_df.loc[celltype2, celltype1] = distance

    ## Convert distance to similarity
    maxd, _ = find_largest_distance(tree)
    pairwise_comparisons_df = (maxd - pairwise_comparisons_df) / maxd

    # Make the DataFrame symmetric by copying the upper triangle to the lower triangle
    pairwise_comparisons_df = pairwise_comparisons_df.astype(float)
    pairwise_comparisons_df = pairwise_comparisons_df.where(~pairwise_comparisons_df.isnull(), pairwise_comparisons_df.T)
    pairwise_comparisons_df.fillna(1, inplace=True)

    reference_celltypes = ['Astrocytes', 'Endothelial', 'Excitatory', 'Inhibitory', 'Microglia', 'OPCs', 'Oligodendrocytes', 'Pericytes']
    celltype_mapping = {celltype: reference_celltypes[i] for i, celltype in enumerate(celltypes)}
    pairwise_comparisons_df = pairwise_comparisons_df.rename(columns=celltype_mapping, index=celltype_mapping)

    plt.figure(figsize=(5, 4))
    sns.heatmap(pairwise_comparisons_df, annot=False, cmap='magma', cbar_kws={'label': 'normalized similarity'})

    return pairwise_comparisons_df, celltype_mapping

filenames = ['dend_multiple_cortical_human.json', 'dend_mtg_human.json', 'dend_ctx_hpc_mouse.json']

sim_matrix_dict = {}
celltype_map_dict = {}

for filename in filenames:
    sim_matrix, celltype_mapping = get_celltype_similarity_matrix(filename)

    filename_no_ext = filename.split('.')[0]
    sim_matrix_dict[filename_no_ext] = sim_matrix
    celltype_map_dict[filename_no_ext] = celltype_mapping


dicts_to_save = [
    sim_matrix_dict,
    celltype_map_dict
]

import pickle
with open(os.path.join(os.environ['DATAPATH'], 'biccn_dend_to_matrix.pkl'), 'wb') as f:
    pickle.dump(dicts_to_save, f)



