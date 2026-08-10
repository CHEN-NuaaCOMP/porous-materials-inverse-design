# Response-conditioned inverse design of disordered aluminium foams

This anonymized reviewer release contains the executable code needed to inspect
and test the manuscript workflow. It includes data alignment, the forward
response predictor, the conditional pore-set generator, deterministic candidate
repair, response-aware screening, geometric parameter export, and reviewer-
requested evaluation scripts. Generated outputs, training histories, figures,
manuscript drafts, local paths, and archived implementations are intentionally
excluded.

## Repository contents

```text
configs/reviewer.yaml          portable frozen configuration
src/                           generator, repair, screening, and reconstruction code
external_predictor/            PredictNet model, preprocessing, training, and weights
analysis/                      reviewer-requested quantitative checks
checkpoints/                   frozen generator and forward-model weights
examples/                      one target curve and one pore-cloud input
tests/test_smoke.py            shape, masking, normalization, and export checks
docs/                          equations, data contract, and experiment commands
```

Two forward-model checkpoints are named explicitly because they serve different
roles:

- `ckpt_base_v1` is the base forward predictor used for held-out predictor
  evaluation.
- `ckpt_phase_terminal_v1` is the phase-weighted predictor used by the frozen
  inverse-design screening pipeline.

Do not combine metrics from these checkpoints without naming the checkpoint.

## Installation

Python 3.10 or 3.11 is recommended. Install a PyTorch build appropriate for the
available CPU or CUDA runtime, then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

For voxel or STL reconstruction, additionally run:

```bash
pip install -r requirements-reconstruction.txt
```

## Quick verification

Run from the repository root:

```bash
python tests/test_smoke.py
python -m compileall -q src external_predictor analysis tests
```

## Minimal end-to-end example

The following commands write all generated files under `outputs/`, which is
ignored by Git:

```bash
python -m src.generate_candidates --config configs/reviewer.yaml --target-curve examples/target_curve.csv --samples 8 --output outputs/demo/candidates
python -m src.repair_candidates --input outputs/demo/candidates/candidates.jsonl --output outputs/demo/repaired/candidates_repaired.jsonl --redistribute --match-macro-porosity --boundary-policy allow_truncated --iterations 10
python -m src.predict_candidate_responses --config configs/reviewer.yaml --input outputs/demo/repaired/candidates_repaired.jsonl --target-curve examples/target_curve.csv --output-dir outputs/demo/predictions
python -m src.screen_candidates --config configs/reviewer.yaml --input outputs/demo/repaired/candidates_repaired.jsonl --target-curve examples/target_curve.csv --prediction-dir outputs/demo/predictions --output-dir outputs/demo/screened
```

## Full dataset contract

The raw scientific dataset is not duplicated in this code repository. Place it
under the following paths before training or full held-out evaluation:

```text
data/full/global.csv
data/full/pore_cloud/<sample>.csv
data/full/stress_strain_responses/<sample>.csv
```

`global.csv` must contain a `name` column and the macroscopic descriptors used
by `src/utils/normalization.py`. The three directories must share the same
sample names. Numeric filename prefixes are identifiers only; every sample is
treated as an independent observation, and prefixes are not used as parent or
family labels.

The generator split is frozen in
`checkpoints/geoloop_gp_v1/split_names.json`. A portable preparation utility for
the separate PredictNet layout is available as:

```bash
python -m src.data.prepare_predictor_dataset --feature-root PATH/TO/FEATURES --curve-dir PATH/TO/CURVES --output-root data/predictnet --split-file PATH/TO/SPLIT.json
```

## Curve-grid interface

The generator conditions on a 50-point interpolation of the complete response
curve. PredictNet retains its trained 51-output head, including the deterministic
zero-strain origin. The adapter constructs the corresponding 51-point evaluation
grid, and screening compares target and prediction on a common grid. This is an
explicit module interface, not a checkpoint-shape mismatch.

## Reviewer experiments

Commands for forward-model evaluation, best-of-K analysis, response-guidance
ablation, structural diversity, retrieval baseline, reconstructed-FE curve
audit, and physical-test scaling are listed in
[`docs/REVIEWER_EXPERIMENTS.md`](docs/REVIEWER_EXPERIMENTS.md).
Mathematical definitions and architecture details are in
[`docs/METHODS_AND_EQUATIONS.md`](docs/METHODS_AND_EQUATIONS.md).

## Scope boundary

This repository can reproduce the machine-learning and geometry-export stages
when the corresponding data are supplied. It does not contain an Abaqus analysis
deck with material, contact, boundary-condition, or step definitions. Therefore,
the included reconstruction code must not be presented as a complete FE solver
reproduction package.

