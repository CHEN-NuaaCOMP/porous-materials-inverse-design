# Response-function-conditioned inverse design enables disorder as a design dimension in porous materials


Code and processed data accompanying:

**Response-function-conditioned inverse design enables disorder as a design dimension in porous materials**

Wei Zhao, Ran Chen, Jiaxiu Zhang, Liangliang Nan, Song Wang, Honghua Chen,

Hanyang Yu, Songcan Chen, and Mingqiang Wei.

*npj Computational Materials* (2026). Published online: 16 October 2026.

**Paper:** [https://doi.org/10.1038/s41524-026-02367-5](https://doi.org/10.1038/s41524-026-02367-5)

This repository contains the manuscript-aligned implementation of the complete
computational workflow: pore-cloud data preparation, full-curve forward
prediction, response-conditioned pore-set generation, candidate repair and
screening, gradient-based geometric refinement, and reconstruction export.

All response curves are represented on 50 equally spaced nominal-strain nodes
over 0--0.6. Sample filenames are unique identifiers; numeric prefixes are not
used as grouping variables.

## Repository layout

```text
configs/manuscript.yaml                 manuscript configuration
external_predictor/                     PredictNet model and training code
src/models/poregen_slot_cvae.py         response-conditioned set generator
src/data/prepare_predictor_dataset.py   deterministic data preparation
src/train.py                            generator training
src/generate_candidates.py              conditional sampling
src/repair_candidates.py                boundary, overlap and porosity repair
src/predict_candidate_responses.py      frozen-surrogate response prediction
src/screen_candidates.py                geometry and response screening
src/optimize_geometry.py                gradient-based geometric refinement
src/export_for_reconstruction.py        reconstruction-table export
checkpoints/                             released PredictNet checkpoint
examples/                                minimal target curve and pore cloud
tests/                                   smoke and manuscript-contract tests
```

## Environment

```bash
conda env create -f environment.yml
conda activate porous-inverse-design
```

Alternatively:

```bash
python -m pip install -r requirements.txt
```


The manuscript implementation used Python 3.10 and PyTorch 2.1.1.

## Data layout

Place the dataset under:

```text
data/full/global.csv
data/full/pore_cloud/<sample>.csv
data/full/stress_strain_responses/<sample>.csv
```

`global.csv` contains one row per sample. Each pore-cloud CSV contains the
centroid coordinates and local pore descriptors. Each response CSV contains
the columns `Strain` and `Stress` and must span nominal strain 0--0.6.

Prepare the fixed PredictNet split and 50-node response grid:

```bash
python -m src.data.prepare_predictor_dataset \
  --feature-root data/full \
  --curve-dir data/full/stress_strain_responses \
  --output-root data/predictnet \
  --split-file splits/predictnet_727_117.json \
  --curve-points 50 \
  --strain-min 0.0 \
  --strain-max 0.6
```

## Forward predictor

The predictor uses parallel local-coordinate and pore-descriptor branches,
global-context fusion, maximum pooling and a 512--256--128--64--50 regression
head. Training-only symmetry augmentation expands 727 base samples to 2,181
instances by adding an x--y reflection and a central reflection; the 117 test
samples remain unchanged.

```bash
python external_predictor/train_single.py \
  --dataset data/predictnet \
  --ckpt outputs/checkpoints/predictnet \
  --result-path outputs/predictnet_results \
  --epochs 200 \
  --selection-mode fixed-final \
  --label-mode global_log1p_standard \
  --augment \
  --drop 0 \
  --jitter 0 \
  --outlier-ratio 0 \
  --device cuda
```

The released checkpoint is stored in
`checkpoints/predictnet_manuscript_8_16/`.

## Response-conditioned generator

```bash
python -m src.train --config configs/manuscript.yaml --device cuda
```

The condition encoder processes the complete 50-node curve, and the masked
set encoder processes ten local pore descriptors during training. A
128-dimensional conditional latent variable is decoded through learned slot
queries and a presence branch, allowing multiple pore configurations to be
sampled for the same target response.

## Generate, repair, predict, screen, refine and export

```bash
python -m src.generate_candidates \
  --config configs/manuscript.yaml \
  --target-curve examples/target_curve.csv \
  --samples 64 \
  --temperature 0.8 \
  --output outputs/demo/candidates

python -m src.repair_candidates \
  --input outputs/demo/candidates/candidates.jsonl \
  --output outputs/demo/repaired/candidates_repaired.jsonl \
  --redistribute \
  --boundary-policy allow_truncated \
  --iterations 80

python -m src.predict_candidate_responses \
  --config configs/manuscript.yaml \
  --input outputs/demo/repaired/candidates_repaired.jsonl \
  --target-curve examples/target_curve.csv \
  --output-dir outputs/demo/predictions \
  --seed 42

python -m src.screen_candidates \
  --config configs/manuscript.yaml \
  --input outputs/demo/repaired/candidates_repaired.jsonl \
  --target-curve examples/target_curve.csv \
  --prediction-dir outputs/demo/predictions \
  --output-dir outputs/demo/screened

python -m src.optimize_geometry \
  --input outputs/demo/screened/accepted_candidates.jsonl \
  --candidate-id candidate_00000_repaired \
  --output-dir outputs/demo/optimized \
  --iterations 200

python -m src.export_for_reconstruction \
  --input outputs/demo/optimized/optimized_candidate.jsonl \
  --candidate-id candidate_00000_repaired_optimized \
  --output outputs/demo/reconstruction/candidate_00000.csv
```

## Verification

```bash
python -m compileall -q src external_predictor analysis tests
python tests/test_smoke.py
python tests/test_manuscript_contract.py
python tests/test_curve_grid.py
python tests/test_release_data.py
python tests/run_synthetic_training_smoke.py --help
```


## Citation

Please cite the associated manuscript when using this code or the released
data.
