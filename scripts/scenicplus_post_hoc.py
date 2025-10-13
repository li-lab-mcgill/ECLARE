#!/usr/bin/env python3
"""
SCENIC+ Downstream Analysis Script

This script implements a complete workflow for downstream analysis of SCENIC+ results
starting from an Excel file containing eGRN data. It rebuilds eRegulons and performs
various downstream analyses including scoring, visualization, and export.

Author: Generated for SCENIC+ downstream analysis
Date: 2024
"""
#%%
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from collections import namedtuple

# SCENIC+ imports
from scenicplus.grn_builder.modules import eRegulon
from scenicplus.scenicplus_class import create_SCENICPLUS_object, mudata_to_scenicplus
from scenicplus.eregulon_enrichment import get_eRegulons_as_signatures, score_eRegulons, binarize_AUC
from scenicplus.RSS import regulon_specificity_scores
from scenicplus.networks import create_nx_graph, create_nx_tables, export_to_cytoscape
from scenicplus.plotting.correlation_plot import correlation_heatmap
from scenicplus.plotting.coverageplot import coverage_plot
from scenicplus.loom import export_to_loom
SCENICPLUS_AVAILABLE = True

# Standard plotting imports
import matplotlib.pyplot as plt
from pathlib import Path
import anndata as ad
import pickle
import seaborn as sns

import scanpy as sc
sc.settings._vector_friendly = True # forces rasterized=True for scatter plots, avoids each scatter point from being a separate SVG vector object

import sys
import os

# Optional: set up your ECLARE env if needed
try:
    sys.path.append("/home/mcb/users/dmannk/scMultiCLIP/ECLARE/src")
    from eclare import set_env_variables
    set_env_variables(config_path='/home/mcb/users/dmannk/scMultiCLIP/ECLARE/config')
except Exception as _e:
    print(f"[INFO] Skipping ECLARE env setup: {_e}")

class SCENICPlusDownstreamAnalyzer:
    """
    A comprehensive class for performing downstream analysis of SCENIC+ results
    starting from Excel data containing eGRN information.
    """

    def __init__(self, excel_path, sheet_name="SCENIC_plus_results"):
        """
        Initialize the analyzer with Excel data.

        Parameters
        ----------
        excel_path : str
            Path to the Excel file containing SCENIC+ results
        sheet_name : str
            Name of the sheet containing the data (default: "SCENIC_plus_results")
        """
        self.excel_path = excel_path
        self.sheet_name = sheet_name
        self.df = None
        self.modules = []
        self.eRegulons = []
        self.scplus_obj = None
        self.gene_signatures = {}
        self.region_signatures = {}

    def load_excel_data(self):
        """Load and validate Excel data."""
        print("Loading Excel data...")
        try:
            self.df = pd.read_excel(self.excel_path, sheet_name=self.sheet_name)
            print(f"Loaded {len(self.df)} rows from '{self.sheet_name}'")
            print(f"Columns: {list(self.df.columns)}")
            return True
        except Exception as e:
            print(f"Error loading Excel file: {e}")
            return False

    def parse_eRegulons_from_excel(self):
        """
        Parse Excel data and group into eRegulon modules.

        Groups rows by (TF, Region_signature_name, Gene_signature_name) triple
        and collects target genes and regions with optional weights.
        """
        if self.df is None:
            print("Error: No data loaded. Call load_excel_data() first.")
            return False

        print("Parsing eRegulons from Excel data...")

        required_cols = ["TF", "Region_signature_name", "Gene_signature_name", "Gene", "Region"]
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            print(f"Warning: Missing required columns: {missing_cols}")
            print(f"Available columns: {list(self.df.columns)}")
            return False

        grp = self.df.groupby(["TF", "Region_signature_name", "Gene_signature_name"], dropna=False)

        self.modules = []
        for (tf, rname, gname), sub in grp:
            genes = sub["Gene"].dropna().astype(str).unique().tolist()
            regions = sub["Region"].dropna().astype(str).unique().tolist()

            if len(genes) == 0 and len(regions) == 0:
                continue

            mod = {
                "tf": str(tf),
                "region_signature": str(rname),
                "gene_signature": str(gname),
                "target_genes": sorted(genes),
                "target_regions": sorted(regions),
            }

            if "TF2G_importance_x_abs_rho" in sub.columns:
                gene_weights = (
                    sub.dropna(subset=["Gene"])
                       .groupby("Gene")["TF2G_importance_x_abs_rho"]
                       .mean()
                       .to_dict()
                )
                mod["gene_weights"] = gene_weights
            else:
                mod["gene_weights"] = {}

            if "R2G_importance_x_abs_rho" in sub.columns:
                region_weights = (
                    sub.dropna(subset=["Region"])
                       .groupby("Region")["R2G_importance_x_abs_rho"]
                       .mean()
                       .to_dict()
                )
                mod["region_weights"] = region_weights
            else:
                mod["region_weights"] = {}

            # Optional is_extended if present
            if "is_extended" in sub.columns:
                # majority vote within the group
                is_ext = bool(pd.Series(sub["is_extended"]).mode(dropna=True)[0]) \
                         if sub["is_extended"].notna().any() else False
            else:
                is_ext = False
            mod["is_extended"] = is_ext

            self.modules.append(mod)

        print(f"Parsed {len(self.modules)} eRegulon modules")
        return True

    def create_eRegulon_objects(self):
        """
        Create SCENIC+ eRegulon objects from parsed modules.
        """
        if not self.modules:
            print("Error: No parsed modules. Call parse_eRegulons_from_excel() first.")
            return False

        if not SCENICPLUS_AVAILABLE:
            print("SCENIC+ not available; creating mock eRegulons instead.")
            self._create_mock_eRegulons()
            return True

        # Define the expected namedtuple for regions2genes entries
        R2G = namedtuple("R2G", ["region", "target", "importance", "rho"])

        # We’ll rebuild regions2genes lists using the original df to carry weights when available
        df = self.df.copy()

        # Make sure numeric columns exist (use defaults otherwise)
        if "TF2G_importance_x_abs_rho" not in df.columns:
            df["TF2G_importance_x_abs_rho"] = 1.0
        if "TF2G_rho" not in df.columns:
            df["TF2G_rho"] = 0.0

        self.eRegulons = []
        # Group either by 3 or 4 keys depending on is_extended availability
        group_keys = ["TF", "Region_signature_name", "Gene_signature_name"]
        if "is_extended" in df.columns:
            group_keys.append("is_extended")

        for keys, sub in df.groupby(group_keys, dropna=False):
            if len(group_keys) == 4:
                tf, r_sig, g_sig, is_ext = keys
                is_ext = bool(is_ext) if pd.notna(is_ext) else False
            else:
                tf, r_sig, g_sig = keys
                is_ext = False

            # Build the R2G list (skip rows missing region or gene)
            r2g_list = []
            sub2 = sub.dropna(subset=["Region", "Gene"]).copy()
            for _, row in sub2.iterrows():
                r2g_list.append(
                    R2G(
                        region=str(row["Region"]),
                        target=str(row["Gene"]),
                        importance=float(row.get("TF2G_importance_x_abs_rho", 1.0)),
                        rho=float(row.get("TF2G_rho", 0.0)),
                    )
                )

            if len(r2g_list) == 0:
                continue

            er = eRegulon(
                transcription_factor=str(tf),
                cistrome_name=str(r_sig),
                is_extended=bool(is_ext),
                regions2genes=r2g_list,
                # you can store the gene signature label as context if useful:
                context=frozenset({str(g_sig)}),
            )

            # If the class doesn't auto-assign a .name, add a readable one
            if not hasattr(er, "name") or er.name is None:
                er.name = f"{tf}::{g_sig}::{r_sig}"

            self.eRegulons.append(er)

        print(f"Created {len(self.eRegulons)} eRegulon objects")
        return True

    def create_scenicplus_object(self, rna_adata, cistopic_obj, key_to_group_by=None):
        """
        Create or load a SCENIC+ object for downstream analysis.
        """
        self.scplus_obj = create_SCENICPLUS_object(
            GEX_anndata=rna_adata,
            cisTopic_obj=cistopic_obj,
            menr={},                        # optional if you only do downstream scoring
            multi_ome_mode=True if key_to_group_by is None else False,            # if True, the function will assume that the RNA and ATAC data are from the same cells
            key_to_group_by=key_to_group_by,    # key used to group unpaired cells
        )
        print("Created SCENIC+ object from AnnData and CistopicObject")


    def register_eRegulons(self):
        """Register eRegulons in the SCENIC+ object and create signatures."""

        print("Registering eRegulons...")

        self.scplus_obj.uns = getattr(self.scplus_obj, "uns", {})
        self.scplus_obj.uns["eRegulons"] = self.eRegulons

        eRegulon_metadata = self.df.copy()
        self.scplus_obj.uns["eRegulon_metadata"] = eRegulon_metadata

        #self.gene_signatures, self.region_signatures = get_eRegulons_as_signatures(self.scplus_obj)
        self.gene_signatures, self.region_signatures = self.df['Gene_signature_name'].unique(), self.df['Region_signature_name'].unique()
        print(f"Created {len(self.gene_signatures)} gene signatures and "
                f"{len(self.region_signatures)} region signatures")
        return True

    def create_mudata_object(self):
        """
        Create MuData object from SCENIC+ object for downstream analysis.
        """
        if self.scplus_obj is None:
            print("Error: No SCENIC+ object. Cannot create MuData.")
            return None
            
        print("Creating MuData object...")
        
        # Import here to avoid circular imports
        from mudata import MuData
        
        # Create the base MuData with RNA and ATAC data
        mudata_obj = MuData({
            "scRNA": ad.AnnData(
                X=self.scplus_obj.to_df('EXP'),
                obs=self.scplus_obj.metadata_cell,
                var=self.scplus_obj.metadata_genes
            ),
            "scATAC": ad.AnnData(
                X=self.scplus_obj.to_df('ACC').T,
                obs=self.scplus_obj.metadata_cell,
                var=self.scplus_obj.metadata_regions
            )
        })
        
        # Add eRegulon AUC data as additional modalities
        if 'eRegulon_AUC' in self.scplus_obj.uns:
            auc_data = self.scplus_obj.uns['eRegulon_AUC']
            idxs = self.scplus_obj.metadata_cell.index
            
            # Add gene-based AUC
            if 'Gene_based' in auc_data:
                mudata_obj.mod['Gene_based'] = ad.AnnData(
                    X=auc_data['Gene_based'].loc[idxs],
                    obs=self.scplus_obj.metadata_cell,
                    var=pd.DataFrame(index=auc_data['Gene_based'].columns)
                )
            
            # Add region-based AUC
            try:
                if 'Region_based' in auc_data:
                    mudata_obj.mod['Region_based'] = ad.AnnData(
                        X=auc_data['Region_based'],
                        bs=self.scplus_obj.metadata_cell,
                        var=pd.DataFrame(index=auc_data['Region_based'].columns)
                    )
            except:
                print("TO DO: Fix addition of region-based AUC")
        
        print(f"Created MuData object with modalities: {list(mudata_obj.mod.keys())}")
        return mudata_obj

    def score_eRegulons_wrapper(self, binarize=True, compute_rss=True, groupby=None):
        """
        Score eRegulons in cells and optionally binarize and compute specificity.
        """

        print("Scoring eRegulons...")
        #import pyranges as pr
        #self.df['Region'] = [pr.PyRanges(row['Region']) for _, row in self.df.iterrows()]

        self.scplus_obj.uns['eRegulon_AUC'] = score_eRegulons(
            self.df, #self.eRegulons,
            gex_mtx=self.scplus_obj.to_df('EXP'),
            acc_mtx=self.scplus_obj.to_df('ACC').T # transpose to match the format of the eRegulon metadata
        )
        
        print("eRegulons scored successfully")

        if binarize:
            binarize_AUC(self.scplus_obj, signature_keys=["Gene_based", "Region_based"])
            print("AUC scores binarized")

        # Create MuData object for regulon_specificity_scores
        self.mudata_obj = self.create_mudata_object()

        if compute_rss:
            rss = regulon_specificity_scores(
                self.mudata_obj,
                variable='scATAC:Age_Range',
                modalities=["scRNA", "scATAC"]
            )
            print(f"Regulon specificity scores computed")
            return rss

        return True

    def _ensure_selected_names(self, selected, fallback_n=10):
        """Utility: produce a list of eRegulon names across real or mock objects."""
        if selected is not None:
            return selected
        if SCENICPLUS_AVAILABLE:
            return [getattr(er, "name", f"{er.transcription_factor}::{er.cistrome_name}")
                    for er in self.eRegulons[:fallback_n]]
        else:
            return [er["name"] for er in self.eRegulons[:fallback_n]]

    def create_network_visualization(self, output_path=os.path.join(os.environ['OUTPATH'], "egrn_cytoscape.cyjs")):
        """
        Create network visualization of eRegulons.
        """

        print("Creating network visualization...")
        nx_tables = create_nx_tables(self.scplus_obj)
        G, pos, edge_tables, node_tables = create_nx_graph(nx_tables)
        export_to_cytoscape(G, pos, out_file=output_path)
        print(f"Network exported to '{output_path}'")


    def create_correlation_heatmap(self, selected_regulons=None, output_path="correlation_heatmap.png"):
        """
        Create correlation heatmap of eRegulon activities.
        """

        print("Creating correlation heatmap...")
        correlation_heatmap(
            self.scplus_obj,
            signature_keys=["Gene_based", "Region_based"],
            selected_regulons=selected_regulons,
        )
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Correlation heatmap saved to '{output_path}'")


    def create_coverage_plot(self, eRegulon_idx=0, groupby="lineage", output_path="coverage_plot.png"):
        """
        Create coverage plot for selected regions.
        """

        # pull regions from real eRegulon
        regions = getattr(self.eRegulons[eRegulon_idx], "target_regions", None)
        region = regions[7]

        coverage_plot(self.scplus_obj, bw_dict=None, region=region)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Coverage plot saved to '{output_path}'")

    def _create_mock_coverage_plot(self, eRegulon_idx, groupby, output_path):
        """Create a mock coverage plot when SCENIC+ is not available."""
        print("Creating mock coverage plot...")
        if eRegulon_idx >= len(self.eRegulons):
            print(f"Error: eRegulon index {eRegulon_idx} out of range")
            return
        regions = self.eRegulons[eRegulon_idx].get("target_regions", [])
        print(f"Mock coverage plot for {len(regions)} regions, grouped by '{groupby}'")
        print(f"Output would be saved to: {output_path}")

    def _deduplicate_regulon_names(self):
        """
        Remove duplicate regulon names from AUC data and metadata to prevent loom export errors.
        """
        if 'eRegulon_AUC' not in self.scplus_obj.uns:
            return
            
        print("Checking for duplicate regulon names...")
        auc_data = self.scplus_obj.uns['eRegulon_AUC']
        
        # Check AUC data
        for signature_key in ['Gene_based', 'Region_based']:
            if signature_key in auc_data:
                df = auc_data[signature_key]
                original_cols = df.columns.tolist()
                
                # Check for duplicates
                if len(original_cols) != len(set(original_cols)):
                    print(f"Found duplicate regulon names in {signature_key}")
                    
                    # Create unique column names by adding suffix for duplicates
                    unique_cols = []
                    col_counts = {}
                    
                    for col in original_cols:
                        if col in col_counts:
                            col_counts[col] += 1
                            unique_cols.append(f"{col}_dup{col_counts[col]}")
                        else:
                            col_counts[col] = 0
                            unique_cols.append(col)
                    
                    # Update the dataframe with unique column names
                    df.columns = unique_cols
                    auc_data[signature_key] = df
                    
                    print(f"Renamed {len(original_cols) - len(set(original_cols))} duplicate columns in {signature_key}")
                else:
                    print(f"No duplicates found in {signature_key}")
        
        # Check and fix eRegulon metadata duplicates
        if 'eRegulon_metadata' in self.scplus_obj.uns:
            metadata = self.scplus_obj.uns['eRegulon_metadata']
            
            # Check for duplicates in gene signature names
            gene_sigs = metadata['Gene_signature_name'].tolist()
            if len(gene_sigs) != len(set(gene_sigs)):
                print(f"Found duplicate gene signature names in metadata")
                
                # Create unique names by adding suffix
                unique_gene_sigs = []
                sig_counts = {}
                
                for sig in gene_sigs:
                    if sig in sig_counts:
                        sig_counts[sig] += 1
                        unique_gene_sigs.append(f"{sig}_dup{sig_counts[sig]}")
                    else:
                        sig_counts[sig] = 0
                        unique_gene_sigs.append(sig)
                
                # Update the metadata
                metadata['Gene_signature_name'] = unique_gene_sigs
                self.scplus_obj.uns['eRegulon_metadata'] = metadata
                
                print(f"Renamed {len(gene_sigs) - len(set(gene_sigs))} duplicate gene signature names")


    def export_to_loom(self, output_path="scenicplus_gene_based.loom"):
        """
        Export results to SCope loom format.
        """
        # Deduplicate regulon names before export
        self._deduplicate_regulon_names()

        print("Exporting to loom format...")
        export_to_loom(self.scplus_obj, signature_key="Gene_based", out_fname=output_path)
        print(f"Results exported to '{output_path}'")
        return True

    def generate_summary_report(self, output_path="scenicplus_summary.txt"):
        """
        Generate a summary report of the analysis.
        """
        print("Generating summary report...")

        with open(output_path, 'w') as f:
            f.write("SCENIC+ Downstream Analysis Summary\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"Input file: {self.excel_path}\n")
            f.write(f"Sheet name: {self.sheet_name}\n")
            f.write(f"Total rows loaded: {len(self.df) if self.df is not None else 0}\n")
            f.write(f"eRegulon modules parsed: {len(self.modules)}\n")
            f.write(f"eRegulon objects created: {len(self.eRegulons)}\n\n")

            f.write("eRegulon Summary:\n")
            f.write("-" * 20 + "\n")
            for i, er in enumerate(self.eRegulons[:10]):  # Show first 10
                if SCENICPLUS_AVAILABLE:
                    name = getattr(er, "name", f"{er.transcription_factor}::{er.cistrome_name}")
                    tf = er.transcription_factor
                    n_genes = len(getattr(er, "target_genes", []))
                    n_regions = len(getattr(er, "target_regions", []))
                else:
                    name = er["name"]
                    tf = er["transcription_factor"]
                    n_genes = len(er["target_genes"])
                    n_regions = len(er["target_regions"])
                f.write(f"{i+1}. {name}\n")
                f.write(f"   TF: {tf}, Genes: {n_genes}, Regions: {n_regions}\n")

            if len(self.eRegulons) > 10:
                f.write(f"... and {len(self.eRegulons) - 10} more eRegulons\n")

        print(f"Summary report saved to '{output_path}'")

def atac_to_cistopic_object(atac):
    from pycisTopic.cistopic_class import create_cistopic_object
    from pycisTopic.lda_models import run_cgs_models, evaluate_models
    from scipy.sparse import csr_matrix

    if 'counts' in atac.layers:
        X_atac = atac.layers['counts']
        region_names = atac.var_names.astype(str).tolist()
    elif atac.raw is not None:
        X_atac = atac.raw.X
        region_names = atac.raw.var_names.astype(str).tolist()
    else:
        raise ValueError("No counts or raw data found in AnnData object")

    fragment_matrix = csr_matrix(X_atac.T)
    fragment_matrix.data = np.rint(fragment_matrix.data).astype(np.int64)
    cell_names = atac.obs_names.astype(str).tolist()

    cistopic_obj = create_cistopic_object(
        fragment_matrix=fragment_matrix,
        cell_names=cell_names,
        region_names=region_names
    )
    cistopic_obj.add_cell_data(atac.obs)

    # Run topic modeling to create the required model attributes
    print("Running cisTopic modeling...")
    models = run_cgs_models(
        cistopic_obj,
        n_topics=[10, 20, 30],  # You can adjust these numbers
        n_iter=100,  # Reduced for speed
        random_state=42
    )

    model = evaluate_models(
        models,
        select_model = 20,
        return_model = True
    )
    
    # Select the best model (e.g., the one with 20 topics)
    cistopic_obj.add_LDA_model(model)
    
    print(f"cisTopic modeling completed. Selected model with {cistopic_obj.selected_model.n_topic} topics.")

    return cistopic_obj

def load_data(source_dataset='Cortex_Velmeshev', target_dataset=None, genes_by_peaks_str='9584_by_66620', valid_cell_ids=None):

    ## modify filename based on presence of target dataset
    if target_dataset is not None:
        rna_filename = f"rna_{genes_by_peaks_str}_aligned_source_{target_dataset}.h5ad"
        atac_filename = f"atac_{genes_by_peaks_str}_aligned_source_{target_dataset}.h5ad"
    else:
        rna_filename = f"rna_{genes_by_peaks_str}.h5ad"
        atac_filename = f"atac_{genes_by_peaks_str}.h5ad"

    ## paths to hard-coded datasets, avoids needing to import from eclare.setup_utils
    if source_dataset == 'MDD':
        rna = ad.read_h5ad(os.path.join(os.environ['DATAPATH'], 'mdd_data', rna_filename), backed='r')
        atac = ad.read_h5ad(os.path.join(os.environ['DATAPATH'], 'mdd_data', atac_filename), backed='r')

    elif source_dataset == 'Cortex_Velmeshev':
        rna = ad.read_h5ad(os.path.join(os.environ['DATAPATH'], source_dataset, "rna", rna_filename), backed='r')
        atac = ad.read_h5ad(os.path.join(os.environ['DATAPATH'], source_dataset, "atac", atac_filename), backed='r')

    ## filter based on valid cell ids
    if valid_cell_ids is not None:
        rna = rna[rna.obs_names.isin(valid_cell_ids)].to_memory()
        atac = atac[atac.obs_names.isin(valid_cell_ids)].to_memory()

    ## ensure that ATAC regions in chrom:start-end format
    atac.var_names = atac.var_names.str.split('[:|-]', expand=True).to_frame().apply(lambda x: f'{x[0]}:{x[1]}-{x[2]}', axis=1).reset_index(drop=True).values
    atac.var_names = atac.var_names.astype(str)

    atac.raw.var.index = atac.raw.var.index.str.split('[:|-]', expand=True).to_frame().apply(lambda x: f'{x[0]}:{x[1]}-{x[2]}', axis=1).reset_index(drop=True).values
    atac.raw.var.index = atac.raw.var.index.astype(str)

    return rna, atac

def add_eregulon_to_dataframe(analyzer, tf_name, target_genes, chromosome_regions, 
                             region_signature_name=None, gene_signature_name=None, 
                             is_extended=False, gene_weights=None, region_weights=None):
    """
    Add a new eRegulon to the analyzer's DataFrame.
    Call this right after parse_eRegulons_from_excel() but before create_eRegulon_objects().
    
    Parameters
    ----------
    analyzer : SCENICPlusDownstreamAnalyzer
        The analyzer object
    tf_name : str
        Name of the transcription factor
    target_genes : list
        List of target gene names
    chromosome_regions : list
        List of chromosome regions
    region_signature_name : str, optional
        Name for the region signature (default: f"{tf_name}_regions")
    gene_signature_name : str, optional
        Name for the gene signature (default: f"{tf_name}_genes")
    is_extended : bool, optional
        Whether this is an extended eRegulon (default: False)
    gene_weights : dict, optional
        Dictionary mapping gene names to weights
    region_weights : dict, optional
        Dictionary mapping region names to weights
    
    Returns
    -------
    bool
        True if successful
    """
    # Set default signature names if not provided
    if region_signature_name is None:
        region_signature_name = f"{tf_name}_regions"
    if gene_signature_name is None:
        gene_signature_name = f"{tf_name}_genes"
    
    # Create default weights if not provided
    if gene_weights is None:
        gene_weights = {gene: 1.0 for gene in target_genes}
    if region_weights is None:
        region_weights = {region: 1.0 for region in chromosome_regions}
    
    print(f"Adding custom eRegulon for TF '{tf_name}' to DataFrame...")
    print(f"  - Target genes: {len(target_genes)}")
    print(f"  - Target regions: {len(chromosome_regions)}")
    print(f"  - Region signature: {region_signature_name}")
    print(f"  - Gene signature: {gene_signature_name}")
    
    # Create new rows for the DataFrame
    new_rows = []
    
    # Create a row for each gene-region pair
    for gene in target_genes:
        for region in chromosome_regions:
            row = {
                'TF': tf_name,
                'Region_signature_name': region_signature_name,
                'Gene_signature_name': gene_signature_name,
                'Gene': gene,
                'Region': region,
                'is_extended': is_extended,
                'TF2G_importance_x_abs_rho': gene_weights.get(gene, 1.0),
                'R2G_importance_x_abs_rho': region_weights.get(region, 1.0),
                'TF2G_rho': 0.0,  # Default correlation
                'R2G_rho': 0.0     # Default correlation
            }
            new_rows.append(row)
    
    # Create DataFrame from new rows
    new_df = pd.DataFrame(new_rows)
    
    # Add to the analyzer's DataFrame
    analyzer.df = pd.concat([analyzer.df, new_df], ignore_index=True)
    
    print(f"Added {len(new_rows)} rows to DataFrame")
    print(f"Total DataFrame size: {len(analyzer.df)} rows")
    
    return True

#%%
if __name__ == "__main__":

    # Load data
    #source_dataset = 'Cortex_Velmeshev'
    #target_dataset = None
    #genes_by_peaks_str = '9584_by_66620'

    #source_dataset = 'MDD'
    #target_dataset = 'PFC_V1_Wang'
    #genes_by_peaks_str = '17279_by_66623'

    source_dataset = 'MDD'
    target_dataset = None
    genes_by_peaks_str = '17563_by_100000'

    # Load ECLARE adata arising from developmental post-hoc analysis
    eclare_adata = ad.read_h5ad(os.path.join(os.environ['OUTPATH'], 'dev_post_hoc_results', f'subsampled_eclare_adata_{source_dataset}.h5ad'))
    rna_adata, atac = load_data(source_dataset=source_dataset, target_dataset=target_dataset, genes_by_peaks_str=genes_by_peaks_str, valid_cell_ids=eclare_adata.obs_names.tolist())

    ## set cell IDs of ATAC to be the same as the RNA
    atac.obs = atac.obs.merge(eclare_adata.obs[['Cell_ID_OT']], left_index=True, right_index=True)
    atac.obs = atac.obs.reset_index().set_index('Cell_ID_OT')

    # Remove duplicate obs_names from rna_adata and atac
    atac = atac[~atac.obs_names.duplicated()].copy()
    rna_adata = rna_adata[rna_adata.obs_names.isin(atac.obs_names)]
 
    ## set cistopic object path
    cistopic_obj_path = os.path.join(os.environ['OUTPATH'], f"cistopic_obj_{source_dataset}_{genes_by_peaks_str if target_dataset is None else genes_by_peaks_str + '_' + target_dataset}.pkl")

    ## train cistopic object
    #cistopic_obj = atac_to_cistopic_object(atac)
    #with open(cistopic_obj_path, 'wb') as f:
    #    pickle.dump(cistopic_obj, f)

    ## load cistopic object
    with open(cistopic_obj_path, 'rb') as f:
        print(f"Loading cistopic_obj from {cistopic_obj_path}")
        cistopic_obj = pickle.load(f)

    ## add suffix to rna_adata obs_names to make consistent with CistopicObject
    rna_adata.obs_names = rna_adata.obs_names + '___cisTopic'

    # Initialize analyzer
    excel_path = os.path.join(os.environ.get('DATAPATH', '.'), "science.adf0834_data_s3.xlsx")
    sheet_name = "SCENIC_plus_results"
    analyzer = SCENICPlusDownstreamAnalyzer(excel_path, sheet_name)

    # Run complete analysis
    print("Starting complete SCENIC+ downstream analysis...")
    
    output_dir = os.path.join(os.environ.get('OUTPATH', '.'), "scenicplus_results")
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    # Step 1: Load and parse data
    if not analyzer.load_excel_data():
        print("Analysis failed at data loading step. Check error messages above.")
        exit(1)
    if not analyzer.parse_eRegulons_from_excel():
        print("Analysis failed at eRegulon parsing step. Check error messages above.")
        exit(1)

    # Add custom eRegulon here (optional)
    add_eregulon_to_dataframe(
        analyzer=analyzer,
        tf_name="NR4A2",
        chromosome_regions=["chr9:71768678-71769178"],
        target_genes=["ABHD17B"],
        region_signature_name="NR4A2_custom_regions",
        gene_signature_name="NR4A2_custom_genes"
    )

    add_eregulon_to_dataframe(
        analyzer=analyzer,
        tf_name="EGR1",
        chromosome_regions=["chr9:71768678-71769178"],
        target_genes=["ABHD17B"],
        region_signature_name="EGR1_custom_regions",
        gene_signature_name="EGR1_custom_genes"
    )

    add_eregulon_to_dataframe(
        analyzer=analyzer,
        tf_name="SOX2",
        chromosome_regions=["chr9:71829361-71829861", "chr9:71830119-71830619"],
        target_genes=["ABHD17B"],
        region_signature_name="SOX2_custom_regions",
        gene_signature_name="SOX2_custom_genes"
    )

    # Step 2: Create eRegulon objects
    if not analyzer.create_eRegulon_objects():
        print("Analysis failed at eRegulon object creation step. Check error messages above.")
        exit(1)

    # Step 3: Create SCENIC+ object (if data available)
    analyzer.create_scenicplus_object(rna_adata, cistopic_obj, key_to_group_by=None)
    analyzer.scplus_obj.dr_cell = {'UMAP': pd.DataFrame(eclare_adata.obsm['X_umap'])}

    # Step 4: Register eRegulons
    analyzer.register_eRegulons()

    # Step 5: Score eRegulons
    analyzer.score_eRegulons_wrapper(compute_rss=False)

    # Step 6: Create visualizations
    analyzer.create_correlation_heatmap(output_path=f"{output_dir}/correlation_heatmap.png")
    analyzer.create_network_visualization(output_path=f"{output_dir}/egrn_cytoscape.cyjs")
    #analyzer.create_coverage_plot(output_path=f"{output_dir}/coverage_plot.png") # requires bigwig files

    # Step 7: Export results
    analyzer.export_to_loom(f"{output_dir}/scenicplus_gene_based.loom")

    # Step 8: Generate summary
    analyzer.generate_summary_report(f"{output_dir}/scenicplus_summary.txt")

    # Save the SCENIC+ object for future use
    scenicplus_obj_path = os.path.join(output_dir, "scenicplus_obj.pkl")
    with open(scenicplus_obj_path, "wb") as f:
        pickle.dump(analyzer.scplus_obj, f)
    print(f"SCENIC+ object saved to '{scenicplus_obj_path}'")

    print(f"Complete analysis finished. Results saved to '{output_dir}'")
    print("Analysis completed successfully!")

    ## extract Region-based and Gene-based eRegulon scores
    eregR = analyzer.scplus_obj.uns['eRegulon_AUC']['Region_based']
    eregG = analyzer.scplus_obj.uns['eRegulon_AUC']['Gene_based']

    ## remove eRegulons with all zeros
    eregR = eregR.loc[:,~np.all(eregR>0, 0)]
    eregG = eregG.loc[:,~np.all(eregG>0, 0)]

    ## remove __cisTopic from eregR and eregG column names
    eregR.index = eregR.index.str.replace('___cisTopic', '')
    eregG.index = eregG.index.str.replace('___cisTopic', '')

    ## revert eregR to Cell_ID rather than Cell_ID_OT
    cell_id_mapper = dict(zip(eclare_adata.obs['Cell_ID_OT'], eclare_adata.obs_names))
    eregR.index = eregR.index.map(cell_id_mapper)

    ## add to eclare_adata
    eclare_adata.uns['eRegulon_AUC_Region_based'] = eregR
    eclare_adata.uns['eRegulon_AUC_Gene_based'] = eregG

    ## save
    #eclare_adata.write_h5ad(os.path.join(os.environ['OUTPATH'], 'dev_post_hoc_results', 'subsampled_eclare_adata.h5ad'))

    ## add to eclare_adata.obs
    eregG.columns = eregG.columns.str.split('_(', regex=False).str[0]
    eregR.columns = eregR.columns.str.split('_(', regex=False).str[0]

    print(f'Removing duplicate columns from eRegulon_AUC_Gene_based and eRegulon_AUC_Region_based')
    eregG = eregG.loc[:, ~eregG.columns.duplicated()]
    eregR = eregR.loc[:, ~eregR.columns.duplicated()]

    ereg = pd.concat([eregG, eregR], axis=0)
        
    ## Rename columns to remove the (Region_based) or (Gene_based) suffix
    ereg_aligned = ereg.copy()
    ereg_aligned.columns = pd.Index([col_aligned[0] for col_aligned in ereg.columns.str.split('_(', regex=False)])
    ereg_merged = ereg_aligned.groupby(ereg_aligned.columns, axis=1).sum()

    import scipy
    ereg_merged_scaled = ereg_merged.copy()
    ereg_merged_scaled = scipy.stats.zscore(ereg_merged, axis=1)
    plt.matshow(ereg_merged_scaled, aspect='auto')

    eclare_adata.obs = eclare_adata.obs.merge(ereg_merged_scaled, left_index=True, right_index=True, how='left')
    #eclare_adata.write_h5ad(os.path.join(os.environ['OUTPATH'], 'dev_post_hoc_results', 'subsampled_eclare_adata_with_eRegulon_scores.h5ad'))

    ## correlate ereg with ordinal_pseudotime
    corrs = eclare_adata.obs.corr(method='spearman')['ordinal_pseudotime']
    corrs = corrs[corrs.index.isin(ereg_merged.columns)]
    most_corr_signature = corrs.index[corrs.argmax()]
    #most_corr_signature_trunc = most_corr_signature.split('_(')[0]
    #most_corr_signatures = ereg.columns[ereg.columns.str.startswith(most_corr_signature_trunc)]

    ## aggregate scores by Lineage
    agg_score = eclare_adata.obs.groupby('Lineage')[ereg_merged.columns].mean()
    max_agg_score_signatures = agg_score.idxmax(axis=1).to_dict()
    sc.pl.draw_graph(eclare_adata, color='Lineage', size=100)
    sc.pl.draw_graph(eclare_adata[eclare_adata.obs['Lineage']=='ExNeu'], color=max_agg_score_signatures['ExNeu'], size=100)
    sc.pl.draw_graph(eclare_adata[eclare_adata.obs['Lineage']=='IN'], color=max_agg_score_signatures['IN'], size=100)

    ## aggregate scores by modality
    agg_score = eclare_adata.obs.groupby('modality')[ereg_merged.columns].mean()
    max_agg_score_signatures = agg_score.idxmax(axis=1).to_dict()
    sc.pl.draw_graph(eclare_adata, color='modality', size=100)
    sc.pl.draw_graph(eclare_adata[eclare_adata.obs['modality']=='RNA'], color=max_agg_score_signatures['RNA'], size=100)
    sc.pl.draw_graph(eclare_adata[eclare_adata.obs['modality']=='ATAC'], color=max_agg_score_signatures['ATAC'], size=100)

    ## aggregate scores by sub_cell_type
    agg_score = eclare_adata.obs.groupby('sub_cell_type')[ereg_merged.columns].mean()
    max_agg_score_signatures = agg_score.idxmax(axis=1).to_dict()
    sc.pl.draw_graph(eclare_adata, color='sub_cell_type', size=100)

    for i, sub_cell_type in enumerate(max_agg_score_signatures.keys()):
        adata_sub_cell_type = eclare_adata.copy()
        adata_sub_cell_type.obs.loc[adata_sub_cell_type.obs['sub_cell_type']!=sub_cell_type, max_agg_score_signatures[sub_cell_type]] = np.nan
        sc.pl.draw_graph(adata_sub_cell_type, color=max_agg_score_signatures[sub_cell_type], size=100)
    
    ## aggregate scores by Age_Range
    agg_score = eclare_adata.obs.groupby('Age_Range')[ereg_merged.columns].mean()
    max_agg_score_signatures = agg_score.idxmax(axis=1).to_dict()

    progenitor_signatures = analyzer.df['lineage']

    rand_idx = np.random.choice(eclare_adata.obs_names, size=len(eclare_adata), replace=False)
    sc.pl.draw_graph(eclare_adata[rand_idx], color=['Age_Range', 'sub_cell_type'], size=100, wspace=0.25)
    for i, age_range in enumerate(max_agg_score_signatures.keys()):
        adata_age_range = eclare_adata.copy()
        adata_age_range.obs.loc[adata_age_range.obs['Age_Range']!=age_range, max_agg_score_signatures[age_range]] = np.nan
        sc.pl.draw_graph(adata_age_range, color=max_agg_score_signatures[age_range], size=100)
        #sc.pl.draw_graph(eclare_adata, color=max_agg_score_signatures[age_range], size=100)
    
    ## aggregate scores by Age_Range and Lineage
    agg_score = eclare_adata.obs.groupby(['Age_Range', 'Lineage'])[ereg_merged.columns].mean()
    max_agg_score_signatures = agg_score.idxmax(axis=1).to_dict()
    for age_range, lineage in max_agg_score_signatures.keys():
        sc.pl.draw_graph(eclare_adata[(eclare_adata.obs['Age_Range']==age_range) & (eclare_adata.obs['Lineage']==lineage)], color=max_agg_score_signatures[age_range, lineage], size=100)
    
    sc.pl.draw_graph(eclare_adata, color=most_corr_signature, size=100)

    ## focus on POU2F1 and DLX5 for VIP sub_cell_type
    #from scipy.special import softmax
    #agg_score = eclare_adata.obs.groupby('sub_cell_type')[ereg_merged.columns].mean()
    #agg_score.columns = agg_score.columns.str.split('_').str[0]
    #softmax_agg_score_vip = agg_score.apply(softmax).loc['VIP']
    #most_distinctive_signatures = softmax_agg_score_vip.sort_values(ascending=False).head(5)

    # Select columns in eclare_adata.obs that are of float dtype
    signatures = eclare_adata.copy()
    signatures.obs = signatures.obs.loc[:, signatures.obs.columns.isin(ereg.columns)]
    signatures.obs.columns = signatures.obs.columns.str.split('_').str[0]
    signatures.obs = signatures.obs.groupby(axis=1, level=0).quantile(0.25) # merge
    signatures.obs = signatures.obs.merge(eclare_adata.obs[['leiden', 'Condition', 'most_common_cluster']], left_index=True, right_index=True, how='left')
    # assert (signatures.obs.groupby(axis=1, level=0).apply(np.max, axis=1) == signatures.obs.groupby(axis=1, level=0).max()).values.mean() > 0.99

    if source_dataset == 'Cortex_Velmeshev':
        # To impose the same scale on the colorbar for both 'POU2F1' and 'DLX5', 
        vmin = signatures.obs[['POU2F1', 'DLX5']].min().min()
        vmax = signatures.obs[['POU2F1', 'DLX5']].max().max()
        
        ## GRNs plots
        fig = sc.pl.draw_graph(signatures, color=['POU2F1', 'DLX5'], size=100, vmin=vmin, vmax=vmax, cmap='plasma', return_fig=True, show=False)
        fig.set_size_inches(10, 4)

        ## VIP density plot
        eclare_adata.obs['is_vip'] = pd.Categorical(eclare_adata.obs['sub_cell_type'] == 'VIP').rename_categories({True: 'VIP', False: 'Non-VIP'})
        sc.tl.embedding_density(eclare_adata, basis='draw_graph_fa', groupby='is_vip')
        vip_fig = sc.pl.embedding_density(eclare_adata, basis='draw_graph_fa', key='draw_graph_fa_density_is_vip', return_fig=True, group='VIP', title='VIP neurons density')
        vip_fig.set_size_inches(4.75, 4)

        ## INT density plot
        eclare_adata.obs['is_INT'] = pd.Categorical(eclare_adata.obs['sub_cell_type'] == 'INT').rename_categories({True: 'INT', False: 'Non-INT'})
        sc.tl.embedding_density(eclare_adata, basis='draw_graph_fa', groupby='is_INT')
        INT_fig = sc.pl.embedding_density(eclare_adata, basis='draw_graph_fa', key='draw_graph_fa_density_is_INT', return_fig=True, group='INT', title='INT neurons density')
        INT_fig.set_size_inches(4.75, 4)

        def dev_fig3(fig, vip_fig, manuscript_figpath=os.path.join(os.environ['OUTPATH'], 'dev_post_hoc_results')):
            fig.savefig(os.path.join(manuscript_figpath, 'dev_fig3_grns.svg'), bbox_inches='tight', dpi=300)
            vip_fig.savefig(os.path.join(manuscript_figpath, 'dev_fig3_vip.svg'), bbox_inches='tight', dpi=300)

    elif source_dataset == 'MDD':
        # To impose the same scale on the colorbar for both 'POU2F1' and 'DLX5', 
        vmin = signatures.obs[['ZNF184', 'NFIX']].min().min()
        vmax = signatures.obs[['ZNF184', 'NFIX']].max().max()
        
        ## GRNs plots
        fig = sc.pl.draw_graph(signatures[eclare_adata.obs['Condition']=='case'], color=['ZNF184', 'NFIX'], groups='case', size=100, vmin=vmin, vmax=vmax, cmap='plasma')#, return_fig=True, show=False)
        fig = sc.pl.draw_graph(signatures[eclare_adata.obs['Condition']=='control'], color=['ZNF184', 'NFIX'], groups='control', size=100, vmin=vmin, vmax=vmax, cmap='plasma')
        #fig.set_size_inches(10, 4)

        sc.pl.draw_graph(signatures, color=['EGR1', 'SOX2', 'NR4A2'], size=100)
        sc.pl.draw_graph(signatures, color=signatures.obs.columns[signatures.obs.columns.isin(['MEF2C', 'SATB2', 'FOXP1', 'POU3F1', 'PKNOX2', 'CUX2', 'THRB', 'POU6F2', 'RORB', 'ZBTB18'])], size=100)  # ['MEF2C', 'SATB2', 'FOXP1', 'POU3F1', 'PKNOX2', 'CUX2', 'THRB', 'POU6F2', 'RORB', 'ZBTB18']

        sns.lineplot(signatures.obs[['leiden','Condition','ZNF184']], x='leiden', y='ZNF184', hue='Condition', errorbar='se', marker='o')
        sns.lineplot(signatures.obs[['leiden','Condition','NFIX']], x='leiden', y='NFIX', hue='Condition', errorbar='se', marker='o')

        ## VIP density plot
        eclare_adata.obs['most_common_cluster'] = pd.Categorical(eclare_adata.obs['most_common_cluster'], categories=eclare_adata.obs.groupby('most_common_cluster')['ordinal_pseudotime'].mean().sort_values(ascending=True).index.tolist(), ordered=True)
        sc.tl.embedding_density(eclare_adata, basis='draw_graph_fa', groupby='most_common_cluster')
        sc.pl.embedding_density(eclare_adata, basis='draw_graph_fa', key='draw_graph_fa_density_most_common_cluster', ncols=5)
        #fig.set_size_inches(4.75, 4)

#%%
from scenicplus.triplet_score import _rank_scores_and_assign_random_ranking_in_range_for_ties, _calculate_cross_species_rank_ratio_with_order_statistics

def calculate_triplet_score(
    eRegulon_metadata: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate triplet score for eRegulons without TF-region importance, since don't have cistrome information.
    /home/mcb/users/dmannk/.conda/envs/scenicplus/lib/python3.11/site-packages/scenicplus/triplet_score.py
    """

    eRegulon_metadata = eRegulon_metadata.copy()

    TF_to_gene_score = eRegulon_metadata["TF2G_importance"].to_numpy()
    region_to_gene_score = eRegulon_metadata["R2G_importance"].to_numpy()

    #rank the scores
    TF_to_gene_rank = _rank_scores_and_assign_random_ranking_in_range_for_ties(
        TF_to_gene_score)
    region_to_gene_rank = _rank_scores_and_assign_random_ranking_in_range_for_ties(
        region_to_gene_score)

    #create rank ratios
    TF_to_gene_rank_ratio = (TF_to_gene_rank.astype(np.float64) + 1) / TF_to_gene_rank.shape[0]
    region_to_gene_rank_ratio = (region_to_gene_rank.astype(np.float64) + 1) / region_to_gene_rank.shape[0]
    
    #create aggregated rank
    rank_ratios = np.array([
        TF_to_gene_rank_ratio, region_to_gene_rank_ratio])
    aggregated_rank = np.zeros((rank_ratios.shape[1],), dtype = np.float64)
    for i in range(rank_ratios.shape[1]):
            aggregated_rank[i] = _calculate_cross_species_rank_ratio_with_order_statistics(rank_ratios[:, i])
    eRegulon_metadata["triplet_rank"] = aggregated_rank.argsort().argsort()
    return eRegulon_metadata

eRegulon_metadata = analyzer.scplus_obj.uns['eRegulon_metadata'].copy()
eRegulon_metadata = calculate_triplet_score(eRegulon_metadata)
triplet_ranks = eRegulon_metadata['triplet_rank']

from pyscenic.binarization import binarize
from typing import Optional, List

def binarize_AUC(scplus_obj,
                auc_key: Optional[str] = 'eRegulon_AUC',
                out_key: Optional[str] = 'eRegulon_AUC_thresholds',
                signature_keys: Optional[List[str]] = ['Gene_based', 'Region_based'],
                n_cpu: Optional[int] = 1):

    """
    Binarize eRegulons using AUCell, ensuring to return binary matrix and thresholds.
    /home/mcb/users/dmannk/other/scenicplus/src/scenicplus/eregulon_enrichment.py
    """

    if not out_key in scplus_obj.uns.keys():
        scplus_obj.uns[out_key] = {}
    for signature in signature_keys:
        auc_mtx = scplus_obj.uns[auc_key][signature]
        auc_mtx_binarized, auc_thresholds = binarize(auc_mtx, num_workers=n_cpu)
        scplus_obj.uns[out_key][signature] = auc_mtx_binarized
        scplus_obj.uns[out_key][signature + '_thresholds'] = auc_thresholds

binarize_AUC(analyzer.scplus_obj, signature_keys=['Gene_based', 'Region_based'])

binary_AUC_gene = analyzer.scplus_obj.uns['eRegulon_AUC_thresholds']['Gene_based'].copy()
binary_AUC_region = analyzer.scplus_obj.uns['eRegulon_AUC_thresholds']['Region_based'].copy()

binary_AUC_gene.index = binary_AUC_gene.index.str.replace('___cisTopic', '')
binary_AUC_region.index = binary_AUC_region.index.str.replace('___cisTopic', '').map(cell_id_mapper)
binary_AUC_df = pd.concat([binary_AUC_gene, binary_AUC_region], axis=0)

binary_AUC_df.columns = binary_AUC_df.columns.str.split('_(', regex=False).str[0]
binary_AUC_df = binary_AUC_df.groupby(binary_AUC_df.columns, axis=1).sum()

# Convert columns of binary_AUC_df to categorical dtype (since binary values)
for col in binary_AUC_df.columns:
    binary_AUC_df[col] = binary_AUC_df[col].astype(int).astype(str).astype('category')

eclare_adata_binarized_AUC = eclare_adata.copy()
eclare_adata_binarized_AUC.obs = eclare_adata_binarized_AUC.obs.drop(columns=ereg_merged.columns)
eclare_adata_binarized_AUC.obs = eclare_adata_binarized_AUC.obs.merge(binary_AUC_df, left_index=True, right_index=True, how='left')

## density plots
sc.tl.embedding_density(eclare_adata_binarized_AUC, basis='draw_graph_fa', groupby=most_corr_signature)
sc.pl.embedding_density(eclare_adata_binarized_AUC, basis='draw_graph_fa', key=f'draw_graph_fa_density_{most_corr_signature}', group='1', title=most_corr_signature)


exit(0)


def create_eregulon_signature(scenicplus_obj, tf_name, target_genes, chromosome_regions, 
                             region_signature_name=None, gene_signature_name=None, 
                             is_extended=False, gene_weights=None, region_weights=None):
    """
    Create an eRegulon signature and add it to the scenicplus_obj.
    
    Parameters
    ----------
    scenicplus_obj : SCENICPLUS object
        The SCENIC+ object to add the eRegulon to
    tf_name : str
        Name of the transcription factor
    target_genes : list
        List of target gene names
    chromosome_regions : list
        List of chromosome regions (e.g., ['chr1:1000-2000', 'chr2:3000-4000'])
    region_signature_name : str, optional
        Name for the region signature (default: f"{tf_name}_regions")
    gene_signature_name : str, optional
        Name for the gene signature (default: f"{tf_name}_genes")
    is_extended : bool, optional
        Whether this is an extended eRegulon (default: False)
    gene_weights : dict, optional
        Dictionary mapping gene names to weights (default: None)
    region_weights : dict, optional
        Dictionary mapping region names to weights (default: None)
    
    Returns
    -------
    eRegulon object
        The created eRegulon object
    """
    from collections import namedtuple
    
    # Set default signature names if not provided
    if region_signature_name is None:
        region_signature_name = f"{tf_name}_regions"
    if gene_signature_name is None:
        gene_signature_name = f"{tf_name}_genes"
    
    # Create default weights if not provided
    if gene_weights is None:
        gene_weights = {gene: 1.0 for gene in target_genes}
    if region_weights is None:
        region_weights = {region: 1.0 for region in chromosome_regions}
    
    # Define the R2G namedtuple for regions2genes entries
    R2G = namedtuple("R2G", ["region", "target", "importance", "rho"])
    
    # Create regions2genes list
    r2g_list = []
    for region in chromosome_regions:
        for gene in target_genes:
            # Get weights, defaulting to 1.0 if not specified
            importance = gene_weights.get(gene, 1.0) * region_weights.get(region, 1.0)
            rho = 0.0  # Default correlation value
            
            r2g_list.append(
                R2G(
                    region=str(region),
                    target=str(gene),
                    importance=float(importance),
                    rho=float(rho)
                )
            )
    
    # Create the eRegulon object
    er = eRegulon(
        transcription_factor=str(tf_name),
        cistrome_name=str(region_signature_name),
        is_extended=bool(is_extended),
        regions2genes=r2g_list,
        context=frozenset({str(gene_signature_name)})
    )
    
    # Set a readable name for the eRegulon
    if not hasattr(er, "name") or er.name is None:
        er.name = f"{tf_name}::{gene_signature_name}::{region_signature_name}"
    
    # Add to scenicplus_obj if it exists
    if scenicplus_obj is not None:
        # Initialize uns if it doesn't exist
        if not hasattr(scenicplus_obj, 'uns') or scenicplus_obj.uns is None:
            scenicplus_obj.uns = {}
        
        # Add to eRegulons list
        if 'eRegulons' not in scenicplus_obj.uns:
            scenicplus_obj.uns['eRegulons'] = []
        
        scenicplus_obj.uns['eRegulons'].append(er)
        
        # Create metadata entry for the eRegulon
        metadata_entry = {
            'TF': tf_name,
            'Region_signature_name': region_signature_name,
            'Gene_signature_name': gene_signature_name,
            'Gene': target_genes,
            'Region': chromosome_regions,
            'is_extended': is_extended
        }
        
        # Add weights if provided
        if gene_weights:
            metadata_entry['TF2G_importance_x_abs_rho'] = [gene_weights.get(gene, 1.0) for gene in target_genes]
        if region_weights:
            metadata_entry['R2G_importance_x_abs_rho'] = [region_weights.get(region, 1.0) for region in chromosome_regions]
        
        # Add to metadata if it exists
        if 'eRegulon_metadata' in scenicplus_obj.uns:
            # Convert to DataFrame if it's not already
            if isinstance(scenicplus_obj.uns['eRegulon_metadata'], dict):
                # Create a new DataFrame from the metadata entry
                new_metadata = pd.DataFrame([metadata_entry])
                scenicplus_obj.uns['eRegulon_metadata'] = pd.concat([
                    scenicplus_obj.uns['eRegulon_metadata'], 
                    new_metadata
                ], ignore_index=True)
            else:
                # It's already a DataFrame, append to it
                new_row = pd.DataFrame([metadata_entry])
                scenicplus_obj.uns['eRegulon_metadata'] = pd.concat([
                    scenicplus_obj.uns['eRegulon_metadata'], 
                    new_row
                ], ignore_index=True)
        else:
            # Create new metadata DataFrame
            scenicplus_obj.uns['eRegulon_metadata'] = pd.DataFrame([metadata_entry])
        
        print(f"Added eRegulon '{er.name}' to scenicplus_obj")
        print(f"  - TF: {tf_name}")
        print(f"  - Target genes: {len(target_genes)}")
        print(f"  - Target regions: {len(chromosome_regions)}")
        print(f"  - Is extended: {is_extended}")
    
    return er


# %%
