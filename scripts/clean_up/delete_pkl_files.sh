#!/bin/bash
# Command to delete the pkl files listed in terminal selection (lines 68-87)
# Run from the directory containing these files

rm -f \
  all_dicts_female.pkl \
  pydeseq2_results_dict.pkl \
  significant_genes_dict.pkl \
  X_rna_dict.pkl \
  X_atac_dict.pkl \
  overlapping_target_genes_dict.pkl \
  overlapping_tfs_dict.pkl \
  scompreg_loglikelihoods_dict.pkl \
  std_errs_dict.pkl \
  tg_expressions_dict.pkl \
  tfrps_dict.pkl \
  tfrp_predictions_dict.pkl \
  slopes_dict.pkl \
  intercepts_dict.pkl \
  intercept_stderrs_dict.pkl \
  enrs_dict.pkl \
  magma_results_dict.pkl \
  baseline_sim_dict.pkl \
  models.pkl \
  valid_cell_ids.pkl

echo "Deleted pkl files"
