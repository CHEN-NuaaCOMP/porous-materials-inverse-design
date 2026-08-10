# Reviewer experiment commands

All commands are run from the repository root. Output paths below are ignored by
Git and may be replaced with any writable directory.

## A. Forward predictor

The prepared PredictNet dataset must contain `train/`, `val/`, `test/`,
`scalers.bin`, and `min_max_vals.bin`.

Evaluate the base checkpoint:

```bash
python analysis/evaluate_forward_checkpoint.py --dataset PATH/TO/PREDICTNET_DATASET --role base --output outputs/reviewer/base_predictor.json
```

Evaluate the phase-terminal checkpoint used for inverse-design screening:

```bash
python analysis/evaluate_forward_checkpoint.py --dataset PATH/TO/PREDICTNET_DATASET --role phase-terminal --output outputs/reviewer/phase_terminal_predictor.json
```

The two JSON files must be reported separately because the label transform and
training objective differ.

## B. End-to-end candidate pipeline

```bash
python -m src.generate_candidates --config configs/reviewer.yaml --target-curve examples/target_curve.csv --samples 64 --output outputs/reviewer/candidates
python -m src.repair_candidates --input outputs/reviewer/candidates/candidates.jsonl --output outputs/reviewer/repaired/candidates_repaired.jsonl --redistribute --match-macro-porosity --boundary-policy allow_truncated --iterations 80
python -m src.predict_candidate_responses --config configs/reviewer.yaml --input outputs/reviewer/repaired/candidates_repaired.jsonl --target-curve examples/target_curve.csv --output-dir outputs/reviewer/predictions
python -m src.screen_candidates --config configs/reviewer.yaml --input outputs/reviewer/repaired/candidates_repaired.jsonl --target-curve examples/target_curve.csv --prediction-dir outputs/reviewer/predictions --output-dir outputs/reviewer/screened
```

This single-target run verifies the interfaces. It is not a replacement for the
fixed held-out target-set evaluation.

## C. Best-of-K and threshold coverage

Given a candidate-level archive with one row per candidate:

```bash
python analysis/evaluate_candidate_archive.py --input PATH/TO/CANDIDATE_STAGE_METRICS.csv --output-dir outputs/reviewer/best_of_k --expected-k 64
```

Report the target count, K, median and 95th percentile of target-wise best-of-K
NRMSE, and the exact definition of every threshold. Do not describe surrogate
threshold coverage as FE or experimental success.

## D. Response-guidance ablation

The input table must contain paired full-model and no-response results for each
seed, with the column names used by the command below:

```bash
python analysis/summarize_response_guidance_ablation.py --input PATH/TO/ABLATION.csv --output outputs/reviewer/response_ablation.json
```

The script reports the seed-wise and mean fold increase after removing response
guidance.

## E. Structural diversity and nearest-training distance

```bash
python analysis/audit_candidate_structural_diversity.py --run-root PATH/TO/FROZEN_TARGET_SET_RUN --dataset-manifest PATH/TO/DATASET_MANIFEST.csv --output-dir outputs/reviewer/diversity
```

This audit uses training-median/IQR-standardized descriptors. It measures
within-target structural diversity and candidate-to-training distance. It does
not infer pairwise response distance when complete candidate response curves are
absent.

## F. Nearest-response retrieval baseline

```bash
python analysis/summarize_generator_baseline.py --input PATH/TO/GENERATOR_VS_RETRIEVAL_TARGETS.csv --output-dir outputs/reviewer/retrieval_baseline
```

Use sample-level target rows. Numeric filename prefixes must not be used as
family labels or correlated groups.

## G. Reconstructed-FE curve audit

The input root must contain one subdirectory per case, with paired `*_gt.csv`
and `*_job.csv` curves where available:

```bash
python analysis/audit_reconstruction_fe.py --input-root PATH/TO/RECONSTRUCTION_FE_CASES --output-dir outputs/reviewer/reconstruction_fe
```

This compares stored curve pairs. It does not recreate an Abaqus solution and
cannot substitute for a material/contact/boundary-condition sensitivity study.

## H. Physical-test scaling audit

The input root must contain `gt/`, `reb/`, and `exp/` subdirectories following
the filenames documented in the script:

```bash
python analysis/physical_validation_loocv.py --input-root PATH/TO/PHYSICAL_CURVES --output-dir outputs/reviewer/physical_loocv
```

The coefficient for each material is fitted on three geometries and evaluated
on the held-out fourth geometry. This avoids fitting and testing the scale on the
same curve.

