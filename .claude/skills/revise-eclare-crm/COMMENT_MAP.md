# ECLARE CRM reviewer comment map

Canonical text lives in
`/home/mcb/users/dmannk/scMultiCLIP/overleaf_eclare/ECLARE_reviewer_comments_overleaf.tex`.
This file is an index, not a substitute. Re-read the tracker subsection
before drafting a response — status macros there are source of truth.

All items start **Open** unless the tracker says otherwise.

## Comment index

| ID | One-line request | Theme | Tier | First places to look |
|---|---|---|---|---|
| **R1.1** | Fair zero-shot vs native-protocol baselines; temper SOTA if staying zero-shot | B | 1 | `sec:clip_benchmark`, `sec:baselines`, `scripts/plot_figures.py`, `scripts/benchmark_*` |
| **R1.2** | Add diagonal methods; state inclusion rules; remove/reclassify MOJITOO | B | 1 | `sec:baselines`, `scripts/benchmark_vertical/mojitoo/`, `scripts/benchmark_diagonal/` |
| **R1.3** | λ sweep + CLIP / KD / OT-CLIP / combo / single- vs multi-teacher ablations | A | 1 | `sec:eclare`, `src/eclare/run_utils.py` (`distil_lambda`), `src/eclare/losses_and_distances_utils.py` (`Knowledge_distillation_fn`) |
| **R1.4** | KD-CLIP beats ECLARE on MDD iLISI; “best balance” framing | G | 3 | `sec:mdd_benchmark`, `scripts/plot_figures.py` (`eclare_mdd`, `kd_clip_mdd`) |
| **R1.5** | Variance / source-identity / n-teachers on a common footing | G | 2 | Fig. 2 point clouds, `src/eclare/post_hoc_utils.py` metric helpers |
| **R1.6a** | Donor-level MDD stats; two-tailed; multiplicity; effect sizes/CIs | F | 1 | `sec:dev`, `scripts/developmental_post_hoc.py`, `scripts/ordinal_post_hoc.py` |
| **R1.6b** | Orthogonal (non-CORAL/teacher) developmental validation | F | 1 | `sec:ordinal_pseudotime`, `scripts/ordinal_scripts/ordinal_run.py` |
| **R1.7** | ABHD17B hub is construction-dependent; keep speculative | F | 2 | `fig6:grn`, `scripts/recreate_vip_egrn_network.py`, `scripts/scenicplus_post_hoc.py` |
| **R1.8** | Report scalability / runtime | E | 2 | `scripts/eclare_scripts/`, `src/eclare/run_utils.py` |
| **R1.9** | Do not select MDD hyperparameters with target ARI/NMI | C | 1 | `src/eclare/eval_utils.py` (`compound_metric`), `src/eclare/tune_utils.py` |
| **R1.m1** | kBET batch = donor/tech batch, not source–target combo | H | 2 | `supfig:batch_kbet`, `sec:eval_metrics` |
| **R1.m2** | Tighten Abstract; split compute vs biology | — | 3 | Abstract in `main.tex` |
| **R1.m3** | Consistent stance on prior-knowledge / GLUE | — | 3 | Introduction + Discussion + `sec:baselines` |
| **R1.m4** | GitHub tutorials: functions, params, expected plots | — | 3 | `README.md`, `REPRODUCE.md` |
| **R2.1** | Assumptions / failure when cell-type composition differs | D | 1 | `sec:eclare`, pairing in `src/eclare/post_hoc_utils.py` (`ot_pairing`) |
| **R2.2** | Modalities beyond RNA+ATAC; tri-modal | — | 3 | Discussion; do not invent tri-modal results |
| **R2.3** | Position vs existing CLIP/contrastive sc methods | — | 3 | Introduction, `sec:baselines` (scCLIP) |
| **R2.4** | Multi-teacher vs more data; same-dataset seed/bootstrap control | A | 1 | Teacher setup in `src/eclare/setup_utils.py` (`teachers_setup`) |
| **R2.5** | Compute cost of many teachers + student | E | 2 | same as R1.8 |
| **R2.6** | No explicit batch-effect model | H | 2 | same as R1.m1 |
| **R2.7** | Validate GRNs in genuinely paired multiome where feasible | F | 2 | `sec:pairing`, `scripts/scenicplus_post_hoc.py` |
| **R3.1** | Weak/noisy teacher ATAC; when teachers limit the student | — | 2 | Teacher CLIP training `scripts/clip_scripts/clip_run.py` |
| **R3.2** | Does n-teachers help monotonically? PFC\_Zhu / DLPFC\_Ma KD-CLIP vs ECLARE | A | 1 | Fig. 2c, Fig. S1b captions; `plot_figures.py` |
| **R3.3** | Rare / unmatched cell types | D | 1 | same as R2.1 |
| **R3.4** | Label-free (or source-only) HPO; Optuna vs defaults; λ | A, C | 1 | `src/eclare/tune_utils.py`, `tab:hyperparameters` |
| **R3.5** | Time complexity vs baselines | E | 2 | same as R1.8 |
| **R3.m1** | What is GT cell type on paired RNA/ATAC? | — | 3 | `src/eclare/setup_utils.py` per-dataset `cell_group` |
| **R3.m2** | Name Fig. 2a source/target configs + implementation detail | B | 3 | `sec:clip_benchmark` benchmarking paragraph, `plot_figures.py` `methods_id_dict` |

Editorial (not numbered): point-by-point letter, STAR Methods, Key Resources
Table, article length, SI formatting, Lead Contact.

## Themes (answer once, cross-reference)

| Theme | IDs | Unified analysis / response |
|---|---|---|
| **A** Multi-teacher + loss ablations | R1.3, R1.4, R2.4, R3.2, R3.4 | n-teachers, teacher subsets, same-dataset multi-teacher control, CLIP / KD / OT-CLIP / KD+OT-CLIP, λ sweep |
| **B** Benchmark fairness + coverage | R1.1, R1.2, R3.m2 | Define zero-shot vs conventional integration; run native-protocol baselines or temper SOTA; expand diagonal methods; drop/reclassify MOJITOO; document source/target combos |
| **C** HPO without target labels | R1.9, R3.4 | Source-domain validation; no target ARI/NMI in MDD model pick; explain Optuna vs frozen defaults |
| **D** Source–target mismatch | R2.1, R3.3 | Drop cell types from source and/or target; rare-cell case; state failure modes |
| **E** Scalability | R1.8, R2.5, R3.5 | Wall-clock train/infer, peak GPU memory, scaling in cells and teachers, vs baselines |
| **F** Biological validation | R1.6b, R1.7, R2.7 | Orthogonal developmental check; paired GRN check; external TF evidence; single-modality DE; keep hypothesis-generating |
| **G** Stability / teacher choice | R1.4, R1.5, R2.4, R3.2 | Per-source metrics, subset performance, CV on comparable combos, when a single teacher wins |
| **H** Batch effects | R1.m1, R2.6 | Clarify kBET labels; donor/technical batch; no dedicated batch-correction loss |

## Suggested order (from the tracker)

**Tier 1** (likely to decide whether the revision is viable):
R1.1 / R1.2, R1.3 / R2.4 / R3.2 / R3.4, R1.9 / R3.4, R1.6a, R1.6b, R2.1 / R3.3.

**Tier 2**: R1.5, R1.7 / R2.7, R1.8 / R2.5 / R3.5, R1.m1 / R2.6, R3.1.

**Tier 3** (mostly prose/docs): R1.4, R1.m2, R1.m3, R1.m4, R2.2, R2.3, R3.m1, R3.m2.

When the user says “next”, pick the first Tier 1 ID whose tracker status is
still Open, preferring a Theme that can close several IDs at once.
