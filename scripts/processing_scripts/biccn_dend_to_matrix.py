from eclare import set_env_variables
set_env_variables(config_path='../../config')

import os
import json
from io import StringIO
from Bio import Phylo
import numpy as np
import seaborn as sns
import pandas as pd
import pickle
from itertools import combinations
import matplotlib.pyplot as plt

import os
import json
import rdflib
from typing import Dict, List, Tuple, Any
import networkx as nx
from tqdm import tqdm

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

def create_biccn_dend_to_matrix():
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

    with open(os.path.join(os.environ['DATAPATH'], 'biccn_dend_to_matrix.pkl'), 'wb') as f:
        pickle.dump(dicts_to_save, f)

def scCello_get_prestored_data(data_file_name: str):

    """
    Load only:
      - "cell_taxonomy_tree_json" -> returns
          (raw_json, edge_list, clid2idx, clid2name)
      - "cell_taxonomy_graph_owl" -> returns
          (edge_list, clid2idx)

    Raises:
      - AssertionError if name unknown or file missing.
      - NotImplementedError for any other key.
    """
    prestored_files = {
        "cell_taxonomy_graph_owl": f"{os.environ['DATAPATH']}/cell_taxonomy/cl.owl",
        "cell_taxonomy_tree_json": f"{os.environ['DATAPATH']}/cell_taxonomy/celltype_relationship.json",

        "cell_taxonomy_biccn_mtg_human_owl": f"{os.environ['DATAPATH']}/cell_taxonomy/dend_mtg_human.owl",
        "cell_taxonomy_biccn_mtg_human_tree_csv": f"{os.environ['DATAPATH']}/cell_taxonomy/biccn_cell_set_nomenclature.csv",

        "cell_taxonomy_biccn_multiple_cortical_human_owl": f"{os.environ['DATAPATH']}/cell_taxonomy/dend_multiple_cortical_human.owl",
        "cell_taxonomy_biccn_multiple_cortical_human_tree_csv": f"{os.environ['DATAPATH']}/cell_taxonomy/biccn_multiple_cortical_human_cell_set_nomenclature.csv",
    }

    assert data_file_name in prestored_files, (
        f"Unsupported data_file_name: {data_file_name}. "
        f"Use 'cell_taxonomy_tree_json' or 'cell_taxonomy_graph_owl'."
    )
    path = prestored_files[data_file_name]
    assert os.path.exists(path), (
        f"{data_file_name} should be at {path}, but does not exist."
    )

    if data_file_name == "cell_taxonomy_tree_json":
        fetched_data: List[Dict[str, Any]] = json.load(open(path))
        ct_tree_edge_list: List[Tuple[int, int]] = []
        ct_tree_vocab_clid2idx: Dict[str, int] = {}
        ct_tree_vocab_clid2name: Dict[str, str] = {}

        for item in fetched_data:
            u_clid = item["id"]
            fa_clid = item["pId"]
            u_name = item["name"]

            if u_clid not in ct_tree_vocab_clid2idx:
                ct_tree_vocab_clid2idx[u_clid] = len(ct_tree_vocab_clid2idx)
            # ensure name consistency
            if u_clid in ct_tree_vocab_clid2name:
                assert ct_tree_vocab_clid2name[u_clid] == u_name
            else:
                ct_tree_vocab_clid2name[u_clid] = u_name

            if fa_clid not in ct_tree_vocab_clid2idx:
                ct_tree_vocab_clid2idx[fa_clid] = len(ct_tree_vocab_clid2idx)

            u_id = ct_tree_vocab_clid2idx[u_clid]
            fa_id = ct_tree_vocab_clid2idx[fa_clid]
            ct_tree_edge_list.append((u_id, fa_id))

        # every CL index has a name except the root marker "#"
        for clid in ct_tree_vocab_clid2idx:
            assert clid in ct_tree_vocab_clid2name or clid == "#", \
                "Missing name while creating cell taxonomy tree"

        return fetched_data, ct_tree_edge_list, ct_tree_vocab_clid2idx, ct_tree_vocab_clid2name

    elif data_file_name == "cell_taxonomy_biccn_mtg_human_tree_csv" or data_file_name == "cell_taxonomy_biccn_multiple_cortical_human_tree_csv":
        df = pd.read_csv(path)
        df['clid'] = 'CL:' + df['cell set accession'].astype(str)
        df.rename(columns={'cell set preferred alias': 'name'}, inplace=True)
        ct_tree_vocab_clid2idx = {row['clid']: row.name for _, row in df.iterrows()}
        ct_tree_vocab_clid2name = {row['clid']: row['name'] for _, row in df.iterrows()}
        ct_tree_edge_list = []
        return df, ct_tree_edge_list, ct_tree_vocab_clid2idx, ct_tree_vocab_clid2name

    elif data_file_name.endswith('owl'):
        g = rdflib.Graph()
        g.parse(path, format="xml")

        raw_edges: List[Tuple[str, str]] = []
        clids: List[str] = []

        for s, _, o in g:
            s_str, o_str = str(s), str(o)
            if "CL" in s_str and "CL" in o_str:
                u_clid = s_str.split("/")[-1].replace("_", ":")
                fa_clid = o_str.split("/")[-1].replace("_", ":")
                if u_clid == fa_clid:
                    continue
                if not (u_clid.startswith("CL:") and fa_clid.startswith("CL:")):
                    continue
                raw_edges.append((u_clid, fa_clid))
                clids.extend([u_clid, fa_clid])

        clid_list = sorted(set(clids))
        ct_graph_vocab_clid2idx: Dict[str, int] = {v: i for i, v in enumerate(clid_list)}
        ct_graph_edge_list: List[Tuple[int, int]] = [
            (ct_graph_vocab_clid2idx[u], ct_graph_vocab_clid2idx[v]) for u, v in raw_edges
        ]

        return ct_graph_edge_list, ct_graph_vocab_clid2idx

    else:
        # Should never hit because of the initial assert, but kept for clarity.
        raise NotImplementedError("Only 'cell_taxonomy_tree_json' and 'cell_taxonomy_graph_owl' are supported.")

    
def scCello_get_cell_taxonomy_similarity(nx_graph, get_raw=False, alpha=0.9, thresh=1e-4):
    """
    Return: a matrix sized by the number of nodes on the cell ontology graph,
        with transformed PPR values (see paper App. A PPR Transformation for details)
    """
    
    # assume node indices ranging from 0
    assert np.max([node_id for node_id in nx_graph.__dict__["_node"].keys()]) + 1 == nx_graph.number_of_nodes()
    num_node = nx_graph.number_of_nodes()
    if get_raw:
        similarity = np.zeros((num_node, num_node), dtype=np.float32)
    else:
        similarity = np.zeros((num_node, num_node), dtype=np.int32)
    for node_id in tqdm(range(num_node), "getting cell taxonomy similarity"):
        personalization = {i: i == node_id for i in range(num_node)}
        ppr = nx.pagerank(nx_graph, alpha=alpha, personalization=personalization)
        for k, v in ppr.items():
            if not get_raw:
                similarity[node_id][k] = 1 if v < thresh else np.log2(v * (1 / thresh) + 1)
            else:
                similarity[node_id][k] = v
    return similarity

def get_scCello_similarity_matrix(filename_key):

    if filename_key == 'dend_mtg_human':
        filename_key_onto = 'cell_taxonomy_biccn_mtg_human_owl'
        filename_key_tree = 'cell_taxonomy_biccn_mtg_human_tree_csv'

    elif filename_key == 'dend_multiple_cortical_human':
        filename_key_onto = 'cell_taxonomy_biccn_multiple_cortical_human_owl'
        filename_key_tree = 'cell_taxonomy_biccn_mtg_human_tree_csv' #'cell_taxonomy_biccn_multiple_cortical_human_tree_csv'

    elif filename_key == 'scCello_cell_ontology':
        filename_key_onto = 'cell_taxonomy_graph_owl'
        filename_key_tree = 'cell_taxonomy_tree_json'

    # get graph similarity
    ct_graph_edge_list, ct_graph_vocab_clid2idx = scCello_get_prestored_data(filename_key_onto)
    _, _, ct_tree_vocab_clid2idx, ct_tree_vocab_clid2name = scCello_get_prestored_data(filename_key_tree)

    ct_graph = nx.Graph(ct_graph_edge_list)
    similarity = scCello_get_cell_taxonomy_similarity(ct_graph, get_raw=False, alpha=0.9)

    if 'dend' in filename_key: # dendrogram
        #ct_edges_df = nx.to_pandas_edgelist(ct_graph)  # seems to symmetrize the graph
        ct_edges_df = pd.DataFrame(ct_graph_edge_list, columns=['source', 'target'])
        assert ct_edges_df['source'].value_counts().unique().item() == 1
        assert ct_edges_df['target'].value_counts().unique().item() == 2

    clid2name_df = pd.DataFrame(ct_tree_vocab_clid2name.items(), columns=['clid', 'name'])
    clid2idx_df = pd.DataFrame(ct_tree_vocab_clid2idx.items(), columns=['clid', 'idx'])
    #clid2idx_df = pd.DataFrame(ct_graph_vocab_clid2idx.items(), columns=['clid', 'idx'])
    nodes_df = pd.merge(clid2name_df, clid2idx_df, left_on='clid', right_on='clid').set_index('idx')

    nodes_order = list(ct_graph.nodes)
    nodes_df = nodes_df.loc[nodes_order]

    ## scCello similarity matrix with cell type labels from nomenclature
    similarity_df = pd.DataFrame(
        similarity,
        index=nodes_df['name'],
        columns=nodes_df['name']
    )

    ## add filename to similarity_df as metadata
    similarity_df.attrs['filename'] = filename_key

    ## save cell type names in file, to be processed by ChatGPT
    ## ChatGPT prompt: Here is a text file containing names of cell types that I would like to map onto a list of reference cell types. Based on the following list of reference cell types, provide a 1-to-1 mapping: ['Excitatory', 'Oligodendrocytes', 'Astrocytes', 'Inhibitory', 'OPCs', 'Microglia', 'Endothelial', 'Pericytes']. Note that the reference celltypes are in brain tissue, so make sure to map only brain cell types together.
    nodes_df['name'].to_csv(os.path.join(os.environ['DATAPATH'], "cell_taxonomy", f"{filename_key}_celltype_names.txt"), index=False)

    return similarity_df, nodes_df

def get_scCello_grouped_similarity_matrix(similarity_df):

    filename_key = similarity_df.attrs['filename']

    ## mapping generated with ChatGPT to map nomenclature to reference types
    brain_celltype_mapping = pd.read_csv(os.path.join(os.environ['DATAPATH'], "cell_taxonomy", f"brain_celltype_mapping_{filename_key}.csv"))

    ## filter to only include cell types in reference types
    similarity_filt_df = similarity_df.loc[
        similarity_df.index.isin(brain_celltype_mapping['Original Cell Name'].values),
        similarity_df.columns.isin(brain_celltype_mapping['Original Cell Name'].values)
    ]

    ## rename cell types to reference types
    similarity_mapped_df = similarity_filt_df.rename(
        columns=brain_celltype_mapping.set_index('Original Cell Name').to_dict()['Mapped Reference Type'],
        index=brain_celltype_mapping.set_index('Original Cell Name').to_dict()['Mapped Reference Type']
    )

    # Group the rows and columns by mapped reference type and compute the mean for each group
    grouped_similarity = similarity_mapped_df.groupby(level=0).mean()
    grouped_similarity = grouped_similarity.groupby(axis=1, level=0).mean()

    # Symmetric normalization (D^(-1/2) * S * D^(-1/2))
    diagonal_values = np.diag(grouped_similarity)
    diagonal_sqrt = np.sqrt(diagonal_values)
    diagonal_sqrt_inv = 1.0 / diagonal_sqrt

    # Handle any zero diagonal values
    diagonal_sqrt_inv[diagonal_sqrt == 0] = 0

    # Create diagonal matrices
    D_sqrt_inv = np.diag(diagonal_sqrt_inv)
    grouped_similarity_normalized = D_sqrt_inv @ grouped_similarity @ D_sqrt_inv

    ## Ensure symmetry
    grouped_similarity_normalized = (grouped_similarity_normalized + grouped_similarity_normalized.T) / 2

    grouped_similarity_normalized.columns = grouped_similarity.columns.astype(str)
    grouped_similarity_normalized.index = grouped_similarity.index.astype(str)

    grouped_similarity_normalized.columns.name = 'cell types'
    grouped_similarity_normalized.index.name = 'cell types'

    # Ensure diagonal is exactly 1
    #grouped_similarity_normalized.iloc[np.eye(grouped_similarity_normalized.shape[0], dtype=bool)] = 1.0

    plt.figure(figsize=(5, 5))
    sns.heatmap(grouped_similarity_normalized, cmap='PuBuGn')

    return grouped_similarity_normalized

## loop over all datasets
filename_keys = ['dend_mtg_human', 'dend_multiple_cortical_human', 'scCello_cell_ontology']
scCello_PPR_similarity_dict = {}

for filename_key in filename_keys:
    similarity_df, nodes_df = get_scCello_similarity_matrix(filename_key)
    grouped_similarity_normalized = get_scCello_grouped_similarity_matrix(similarity_df)
    scCello_PPR_similarity_dict[filename_key] = grouped_similarity_normalized

## aggregate scCello PPR similarity
agg_scCello_PPR_similarity = pd.concat(scCello_PPR_similarity_dict.values()).groupby(level=0).mean()
agg_scCello_PPR_similarity_zero_diag = agg_scCello_PPR_similarity.copy()
np.fill_diagonal(agg_scCello_PPR_similarity_zero_diag.values, 0)

scCello_PPR_similarity_dict['agg_scCello_PPR_similarity'] = agg_scCello_PPR_similarity
scCello_PPR_similarity_dict['agg_scCello_PPR_similarity_zero_diag'] = agg_scCello_PPR_similarity_zero_diag

plt.figure(figsize=(5, 5))
sns.heatmap(agg_scCello_PPR_similarity, cmap='PuBuGn')

plt.figure(figsize=(5, 5))
sns.heatmap(agg_scCello_PPR_similarity_zero_diag, cmap='PuBuGn')

## save scCello PPR similarity
with open(os.path.join(os.environ['DATAPATH'], "cell_taxonomy", "scCello_PPR_similarity.pkl"), 'wb') as f:
    pickle.dump(scCello_PPR_similarity_dict, f)

import json
from pathlib import Path
from rdflib import Graph, Namespace, RDF, RDFS, OWL, Literal, URIRef

def json_dendrogram_to_owl(json_path: str, owl_out: str):

    BASE = Namespace("http://purl.obolibrary.org/obo/")  # ensures last path seg is 'CL_*'

    def class_iri(local: str) -> URIRef:
        # local like 'CL_CS202204130_1' -> http://.../CL_CS202204130_1
        return BASE[local]

    def best_label(rec: dict) -> str:
        return (
            rec.get("label")
            or rec.get("original_label")
            or rec.get("cell_set_label")
            or rec.get("cell_set_preferred_alias")
            or rec.get("cell_set_aligned_alias")
            or rec.get("cell_set_additional_aliases")
            or rec.get("_row")
            or ""
        )

    def add_class(g: Graph, acc: str, label: str):
        iri = class_iri(f"CL_{acc}")
        g.add((iri, RDF.type, OWL.Class))
        if label:
            g.add((iri, RDFS.label, Literal(label)))
        return iri

    def walk(node: dict, g: Graph, parent_acc: str):
        # internal node
        if "node_attributes" in node:
            rec = node["node_attributes"][0]
            acc = rec["cell_set_accession"]  # e.g., CS202204130_129
            lbl = best_label(rec)
            cur_iri = add_class(g, acc, lbl)
            if parent_acc is not None:
                g.add((cur_iri, RDFS.subClassOf, class_iri(f"CL_{parent_acc}")))
            for ch in node.get("children", []):
                walk(ch, g, acc)

        # leaves directly under this node
        if "leaf_attributes" in node:
            for leaf in node["leaf_attributes"]:
                acc = leaf["cell_set_accession"]
                lbl = best_label(leaf)
                leaf_iri = add_class(g, acc, lbl)
                if parent_acc is not None:
                    g.add((leaf_iri, RDFS.subClassOf, class_iri(f"CL_{parent_acc}")))

    g = Graph()
    g.bind("owl", OWL)
    g.bind("rdfs", RDFS)

    # Add a synthetic root so every node has a CL:* parent, satisfying your loader
    add_class(g, "ROOT", "All cells")

    data = json.load(open(json_path))
    # Some dumps have a single root dict; others wrap in a top-level object with fields.
    # If yours is a dict with 'children', iterate those; otherwise assume it's the root dict.
    root_nodes = data.get("children", [data]) if isinstance(data, dict) else [data]
    for root in root_nodes:
        walk(root, g, "ROOT")

    g.serialize(owl_out, format="xml")

# Example:
# json_path = os.path.join(os.environ['DATAPATH'], "cell_taxonomy", "dend_multiple_cortical_human.json")
# owl_out = os.path.join(os.environ['DATAPATH'], "cell_taxonomy", "dend_multiple_cortical_human.owl")
# json_dendrogram_to_owl(json_path, owl_out)
