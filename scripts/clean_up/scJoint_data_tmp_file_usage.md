# scJoint_data_tmp File Usage Analysis

## Summary

**Only ONE file is loaded after scJoint training completes:**
- `{timestamp}/scJoint_latents.h5ad` - This is the ONLY file read by `developmental_post_hoc.py`

All other files in `scJoint_data_tmp` are **intermediate files** used only during training and can be safely deleted after training completes.

## File Usage Breakdown

### Files in Root Directory (Intermediate - Can be deleted after training)

These files are created and used **during** scJoint training but are **NOT** read after training completes:

1. **`.h5` files** (e.g., `rna_source_*.h5`, `atac_source_*.h5`)
   - **Created by:** `write_10X_h5()` function in `scJoint_latents.py` (line 153-154)
   - **Used by:** `process_db.data_parsing()` to convert to `.npz` format (line 162)
   - **Status:** Intermediate file, deleted after conversion

2. **`.csv` files** (e.g., `*_celltypes.csv`)
   - **Created by:** `write_10X_h5()` function (saves celltype metadata)
   - **Used by:** `process_db.label_parsing()` to convert to `.txt` format (lines 164, 166)
   - **Status:** Intermediate file, converted to `.txt` format

3. **`.npz` files** (e.g., `rna_source_*.npz`, `atac_source_*.npz`)
   - **Created by:** `process_db.data_parsing()` from `.h5` files (line 27 in process_db.py)
   - **Used by:** scJoint training (loaded via config.rna_paths and config.atac_paths, lines 176-179)
   - **Status:** Intermediate file, used during training only

4. **`.txt` files (celltypes)** (e.g., `*_celltypes.txt`)
   - **Created by:** `process_db.label_parsing()` from `.csv` files (lines 69, 81 in process_db.py)
   - **Used by:** scJoint training (loaded via config.rna_labels and config.atac_labels, lines 177-179)
   - **Status:** Intermediate file, used during training only

5. **`label_to_idx.txt`**
   - **Created by:** `process_db.label_parsing()` (line 58 in process_db.py)
   - **Used by:** scJoint training for label mapping
   - **Status:** Intermediate file, used during training only

### Files in Timestamped Subdirectories (e.g., `20250903_183146/`)

These are created during training in subdirectories named by timestamp:

6. **`*_embeddings.txt`** (e.g., `rna_source_*_embeddings.txt`, `atac_source_*_embeddings.txt`)
   - **Created by:** scJoint training process (`write_embeddings()` in trainingprocess_stage3.py)
   - **Used by:** `scJoint_latents.py` to create final `scJoint_latents.h5ad` (lines 226-227)
   - **Status:** Intermediate file, used to create final h5ad, then can be deleted

7. **`*_predictions.txt`** (e.g., `rna_source_*_predictions.txt`)
   - **Created by:** scJoint training process (optional output)
   - **Used by:** Not used by ECLARE scripts
   - **Status:** Optional intermediate file, can be deleted

8. **`scJoint_latents.h5ad`** ⭐ **ONLY FILE LOADED AFTER TRAINING**
   - **Created by:** `scJoint_latents.py` (line 253)
   - **Used by:** `developmental_post_hoc.py` (line 3214-3215)
   - **Status:** **KEEP THIS FILE** - This is the only file needed after training

## Recommendation

### Safe to Delete:
- All files in the **root** of `scJoint_data_tmp/` (`.h5`, `.npz`, `.csv`, `.txt`, `label_to_idx.txt`)
- All `*_embeddings.txt` and `*_predictions.txt` files in timestamped subdirectories
- Entire timestamped subdirectories **EXCEPT** the `scJoint_latents.h5ad` file

### Must Keep:
- `{timestamp}/scJoint_latents.h5ad` files in each timestamped subdirectory that you want to use for analysis

## Cleanup Script Suggestion

You could safely delete all intermediate files while keeping only the final outputs:

```bash
# Keep only scJoint_latents.h5ad files, delete everything else
find /home/mcb/users/dmannk/scMultiCLIP/outputs/scJoint_data_tmp/ \
  -type f ! -name "scJoint_latents.h5ad" -delete

# Delete empty directories
find /home/mcb/users/dmannk/scMultiCLIP/outputs/scJoint_data_tmp/ \
  -type d -empty -delete
```

**Note:** Only run cleanup if you're certain training is complete and you don't need to re-run scJoint with the same intermediate files.
