#!/usr/bin/env python3
"""
SCENIC+ Downstream Analysis Script

This script implements a complete workflow for downstream analysis of SCENIC+ results
starting from an Excel file containing eGRN data. It rebuilds eRegulons and performs
various downstream analyses including scoring, visualization, and export.

Author: Generated for SCENIC+ downstream analysis
Date: 2024
"""

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

    def create_scenicplus_object(self, rna_adata=None, cistopic_obj=None):
        """
        Create or load a SCENIC+ object for downstream analysis.
        """
        self.scplus_obj = create_SCENICPLUS_object(
            GEX_anndata=rna_adata,
            cisTopic_obj=cistopic_obj,
            menr={},                        # optional if you only do downstream scoring
            multi_ome_mode=True,            # if True, the function will assume that the RNA and ATAC data are from the same cells
            #key_to_group_by=None,    # key used to group unpaired cells
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
            
            # Add gene-based AUC
            if 'Gene_based' in auc_data:
                mudata_obj.mod['Gene_based'] = ad.AnnData(
                    X=auc_data['Gene_based'],
                    obs=self.scplus_obj.metadata_cell,
                    var=pd.DataFrame(index=auc_data['Gene_based'].columns)
                )
            
            # Add region-based AUC  
            if 'Region_based' in auc_data:
                mudata_obj.mod['Region_based'] = ad.AnnData(
                    X=auc_data['Region_based'].T,
                    obs=self.scplus_obj.metadata_cell,
                    var=pd.DataFrame(index=auc_data['Region_based'].columns)
                )
        
        print(f"Created MuData object with modalities: {list(mudata_obj.mod.keys())}")
        return mudata_obj

    def score_eRegulons_wrapper(self, binarize=True, compute_rss=True, groupby=None):
        """
        Score eRegulons in cells and optionally binarize and compute specificity.
        """

        print("Scoring eRegulons...")

        self.scplus_obj.uns['eRegulon_AUC'] = score_eRegulons(
            self.df, #self.eRegulons,
            gex_mtx=self.scplus_obj.to_df('EXP'),
            acc_mtx=self.scplus_obj.to_df('ACC')
        )
        
        sys.stdout.flush()
        print("eRegulons scored successfully")

        if binarize:
            binarize_AUC(self.scplus_obj, signature_keys=["Gene_based", "Region_based"])
            print("AUC scores binarized")

        # Create MuData object for regulon_specificity_scores
        self.mudata_obj = self.create_mudata_object()

        if compute_rss and groupby is not None:
            rss = regulon_specificity_scores(
                self.mudata_obj,
                variable='scATAC:GEX_cluster',
                modalities=["scRNA", "scATAC"]
            )
            print(f"Regulon specificity scores computed for '{groupby}'")
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

    def run_complete_analysis(
        self,
        rna_adata=None,
        cistopic_obj=None,
        mudata_obj=None,
        groupby=None,
        output_dir="scenicplus_output",
    ):
        """
        Run the complete downstream analysis pipeline.
        """
        print("Starting complete SCENIC+ downstream analysis...")

        Path(output_dir).mkdir(exist_ok=True, parents=True)

        # Step 1: Load and parse data
        if not self.load_excel_data():
            return False
        if not self.parse_eRegulons_from_excel():
            return False

        # Step 2: Create eRegulon objects
        if not self.create_eRegulon_objects():
            return False

        # Step 3: Create SCENIC+ object (if data available)
        self.create_scenicplus_object(rna_adata, cistopic_obj)

        if self.scplus_obj is not None:
            # Step 4: Register eRegulons
            self.register_eRegulons()

            # Step 5: Score eRegulons
            self.score_eRegulons_wrapper(compute_rss=(groupby is not None), groupby=groupby)

        # Step 6: Create visualizations
        self.create_correlation_heatmap(output_path=f"{output_dir}/correlation_heatmap.png")
        self.create_network_visualization(output_path=f"{output_dir}/egrn_cytoscape.cyjs")
        self.create_coverage_plot(output_path=f"{output_dir}/coverage_plot.png")

        # Step 7: Export results
        if self.scplus_obj is not None:
            self.export_to_loom(f"{output_dir}/scenicplus_gene_based.loom")

        # Step 8: Generate summary
        self.generate_summary_report(f"{output_dir}/scenicplus_summary.txt")

        print(f"Complete analysis finished. Results saved to '{output_dir}'")
        return True


def main(rna_adata, cistopic_obj):
    """
    Example usage of the SCENICPlusDownstreamAnalyzer.
    """
    # Example configuration (update these env vars or replace with absolute paths)
    excel_path = os.path.join(os.environ.get('DATAPATH', '.'), "science.adf0834_data_s3.xlsx")
    sheet_name = "SCENIC_plus_results"

    # Initialize analyzer
    analyzer = SCENICPlusDownstreamAnalyzer(excel_path, sheet_name)

    # Run complete analysis
    success = analyzer.run_complete_analysis(
        rna_adata=rna_adata,
        cistopic_obj=cistopic_obj,
        # mudata_obj=your_mudata,
        groupby="lineage",  # Update with your grouping column if you want RSS
        output_dir=os.path.join(os.environ.get('OUTPATH', '.'), "scenicplus_results"),
    )

    if success:
        print("Analysis completed successfully!")
    else:
        print("Analysis failed. Check error messages above.")

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

def load_data(source_dataset='Cortex_Velmeshev', target_dataset=None, genes_by_peaks_str='9584_by_66620'):
    rna = ad.read_h5ad(os.path.join(os.environ['DATAPATH'], source_dataset, "rna", f"rna_{genes_by_peaks_str}.h5ad"), backed='r')
    atac = ad.read_h5ad(os.path.join(os.environ['DATAPATH'], source_dataset, "atac", f"atac_{genes_by_peaks_str}.h5ad"), backed='r')

    ## Subsample to 1000 cells for prototyping
    rna = rna[:100000].to_memory()
    atac = atac[:100000].to_memory()

    return rna, atac

if __name__ == "__main__":
    ## Load data
    rna_adata, atac = load_data(source_dataset='PFC_V1_Wang', target_dataset=None, genes_by_peaks_str='9914_by_63404')

    rna_adata.obs_names = rna_adata.obs_names + '___cisTopic'
    
    cistopic_obj = atac_to_cistopic_object(atac)
    with open(os.path.join(os.environ['OUTPATH'], 'cistopic_obj.pkl'), 'wb') as f:
        pickle.dump(cistopic_obj, f)
    
    #with open(os.path.join(os.environ['OUTPATH'], 'cistopic_obj.pkl'), 'rb') as f:
    #    cistopic_obj = pickle.load(f)

    ## Run the main function
    main(rna_adata, cistopic_obj)