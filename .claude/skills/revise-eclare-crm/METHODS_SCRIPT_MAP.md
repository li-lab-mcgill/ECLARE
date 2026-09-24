# ECLARE Methods → implementation source map

Use this map when a reviewer comment or `main.tex` Methods subsection needs
to be explained or revised. Read the manuscript passage first, then the
listed anchors. This map locates code; it is not evidence for numerical
claims. Job IDs and metric values must be re-read from scripts/outputs.

## Root aliases

- `OL`: `/home/mcb/users/dmannk/scMultiCLIP/overleaf_eclare`
- `ECLARE`: `/home/mcb/users/dmannk/scMultiCLIP/ECLARE`

## Main-text Methods map

| Methods / Results area (`main.tex`) | Primary implementation | Anchors to read |
|---|---|---|
| **ECLARE framework / total loss** (`sec:eclare`, `eq:total_loss`) | `ECLARE/src/eclare/run_utils.py`, `ECLARE/src/eclare/losses_and_distances_utils.py`, `ECLARE/src/eclare/models.py` | Student loop: `eclare_pass`, `run_ECLARE`. Losses: `Knowledge_distillation_fn.kl_forward`, `ot_clip_loss_forward`, `distil_loss_weighting`. Default `distil_lambda=0.1` logged in `run_ECLARE`. |
| **CLIP teacher training** (`sec:model_training`) | `ECLARE/scripts/clip_scripts/clip_run.py`, `ECLARE/src/eclare/run_utils.py` (`run_CLIP`, `clip_pass`), `ECLARE/src/eclare/losses_and_distances_utils.py` (`clip_loss`) | InfoNCE on paired source. Launchers: `scripts/clip_scripts/clip_*.sh`. |
| **KD-CLIP (single teacher)** | `ECLARE/scripts/kd_clip_scripts/`, `ECLARE/scripts/eclare_scripts/eclare_run.py` | Same student trainer as ECLARE with one teacher. Launchers: `kd_clip_paired_data.sh`, `kd_clip_mdd.sh`, `kd_clip_dev_stages.sh`. |
| **ECLARE (multi-teacher) training** | `ECLARE/scripts/eclare_scripts/eclare_run.py`, `ECLARE/src/eclare/setup_utils.py` (`teachers_setup`) | Launchers: `eclare_paired_data.sh`, `eclare_mdd.sh`, `eclare_dev_stages.sh`. Optuna: `ECLARE/src/eclare/tune_utils.py` (`tune_ECLARE`, `tune_CLIP`). |
| **Hyperparameters table** (`tab:hyperparameters`) | `ECLARE/src/eclare/models.py` (`get_clip_hparams`), `tune_utils.py` | Confirm table against current defaults; do not copy from memory. |
| **Datasets / loaders** (`tab:dataset-summary`) | `ECLARE/src/eclare/setup_utils.py` | Per-dataset `*_setup` functions (`pfc_zhu_setup`, `dlpfc_ma_setup`, `mdd_setup`, `cortex_velmeshev_setup`, …). Cell-type key is the `cell_group` argument (R3.m1). |
| **Evaluation metrics** (`sec:eval_metrics`) | `ECLARE/src/eclare/eval_utils.py` | `align_metrics`, `unpaired_metrics`, `foscttm`, `iLISI`, `compound_metric` (iLISI+NMI+ARI — R1.9). |
| **Fig. 2 benchmark plots** (`fig2:benchmark`, `sec:clip_benchmark`) | `ECLARE/scripts/plot_figures.py`, `ECLARE/src/eclare/post_hoc_utils.py` (`get_metrics`, `combined_plot`) | Provenance job IDs: `methods_id_dict` at top of `plot_figures.py`. |
| **Baseline runs** (`sec:baselines`) | `ECLARE/scripts/benchmark_vertical/` and `benchmark_diagonal/` | Vertical: `mojitoo/mojitoo_run.py`, `multiVI/multiVI_run.py`, `glue/glue_run.py`, `seurat/seurat_run.py`. Diagonal: `scDART/scDART_run.py`, `scJoint/scJoint_run.py`. MOJITOO is vertical (R1.2). |
| **MDD unpaired integration** (`sec:mdd_benchmark`) | `eclare_mdd.sh` / `kd_clip_mdd.sh` / `clip_mdd.sh`, `plot_figures.py` (`*_mdd` keys) | Target is unpaired MDD; teachers from paired sources. |
| **Nucleus pairing / OT** (`sec:pairing`) | `ECLARE/src/eclare/post_hoc_utils.py` (`ot_pairing`, `cell_gap_ot`) | Downstream GRN pairing, not the training OT-CLIP loss. |
| **SEACells / sc-compReg / Enrichr / MAGMA** (`sec:seacells`, `sec:sc-compreg`, `sec:enrichr`) | `ECLARE/src/eclare/post_hoc_utils.py`, `ECLARE/scripts/enrichment_analyses.py`, `ECLARE/scripts/enrichment_plots.py` | `run_SEACells`, `differential_grn_analysis`, `do_enrichr`, `run_magma`, `run_gseapy`. |
| **Pruned GRNs / ABHD17B network** (`sec:pruned_grns`, `fig6:grn`) | `ECLARE/scripts/recreate_vip_egrn_network.py`, `ECLARE/scripts/merge_eqtl_edges.py` | Network is grown from a seed (NR4A2 → ABHD17B) — central to R1.7. |
| **SCENIC+ VIP eRegulons** (`sec:scenic+`) | `ECLARE/scripts/scenicplus_post_hoc.py` | Developmental Fig. 4d / VIP scores. |
| **Ordinal / CORAL pseudotime** (`sec:ordinal_pseudotime`, `sec:dev`) | `ECLARE/scripts/ordinal_scripts/ordinal_run.py`, `ECLARE/src/eclare/run_utils.py` (`run_ORDINAL`, `ordinal_pass`), `ECLARE/scripts/ordinal_post_hoc.py`, `ECLARE/scripts/developmental_post_hoc.py` | Developmental teacher/CORAL dependence is the R1.6b issue. Snapshot tag: `v1.0-fig4-dev` (`REPRODUCE.md`). |
| **Cross-species analyses** | `ECLARE/scripts/cross_species_analysis.py` | Includes a MOJITOO comparison block — not the Fig. 2a protocol. |
| **Code & data availability** | `ECLARE/README.md`, `ECLARE/REPRODUCE.md` | Zenodo DOI in README / availability paragraph. Tutorials = R1.m4. |

## Supplementary analyses

| SI item | Implementation |
|---|---|
| kBET (`supfig:batch_kbet`) | Metric computation in `eval_utils.py` / plotting in `plot_figures.py` — **verify what column is called “batch”** before answering R1.m1 |
| Enrichr / GREAT / H-MAGMA / module scores / ABHD17B expression (`supfig:*`) | `enrichment_analyses.py`, `enrichment_plots.py`, `post_hoc_utils.py` |
| SI Methods (rGREAT, H-MAGMA) | Described in `OL/main.tex` SI block and `OL/supplementary_information.tex`; R implementations are not in the Python package — do not invent scripts |

## Retrieval rules

1. Resolve the comment ID via `COMMENT_MAP.md`, then the manuscript heading/label.
2. Read that passage in `OL/main.tex`.
3. Read the primary script and named function above; follow local imports only when the detail lives there.
4. For job IDs, λ, metrics, and n-teachers, re-read the current file and saved outputs. `plot_figures.py` `methods_id_dict` is the Fig. 2 provenance.
5. If a listed path is missing, report the mismatch — do not substitute a similarly named script.
6. Frozen reproducibility snapshots (`v1.0-fig4-dev`, `v1.0-main`) are for rerunning published figures, not for silently rewriting current `main` training code.
