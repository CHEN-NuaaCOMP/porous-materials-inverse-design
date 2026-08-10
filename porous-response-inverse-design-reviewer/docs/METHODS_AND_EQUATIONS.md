# Code-derived methods and equations

This document records the implementation that accompanies the frozen release.
Symbols, dimensions, and weights are tied to `configs/reviewer.yaml` and the
source files named below.

## 1. Data representation

A target response is represented by a strain-stress sequence
\(c\in\mathbb{R}^{50\times 2}\). A pore set is
\(S=\{s_i\}_{i=1}^{N}\), where each row contains ten normalized descriptors:
three coordinates, equivalent diameter, area, volume, anisotropy, two
orientation angles, and sphericity. Five macroscopic descriptors form
\(g\in\mathbb{R}^{5}\). Variable pore counts are represented by a Boolean mask;
the configured slot capacity is a batching and checkpoint-shape parameter, not
a physical assertion that every specimen has the same pore count.

The 844 aligned samples are independent observations. Numeric filename prefixes
are identifiers and are not used as parent-specimen or family labels. The frozen
generator split contains 675 training, 84 validation, and 85 test samples.

## 2. Conditional pore-set generator

Implementation: `src/models/poregen_slot_cvae.py`.

### Curve and set encoders

The curve encoder applies one-dimensional convolutions
\(2\rightarrow64\rightarrow160\), followed by a bidirectional GRU with 160
hidden units per direction and a normalized projection. The masked set encoder
applies a pointwise MLP \(10\rightarrow192\rightarrow192\), then concatenates
masked mean and masked maximum pooling to obtain a 384-dimensional set
representation. A separate MLP maps the five global descriptors to 96
dimensions.

### Conditional prior and training posterior

The curve representation \(h_c\) parameterizes a diagonal-Gaussian conditional
prior,

\[
p_\phi(z\mid c)=\mathcal{N}\!\left(\mu_p(h_c),
\operatorname{diag}(\sigma_p^2(h_c))\right).
\]

During training, the curve, pore-set, and global representations parameterize
the posterior,

\[
q_\psi(z\mid c,S,g)=\mathcal{N}\!\left(\mu_q(h_c,h_S,h_g),
\operatorname{diag}(\sigma_q^2(h_c,h_S,h_g))\right),
\]

with a 128-dimensional latent variable sampled as

\[
z=\mu_q+\sigma_q\odot\epsilon,\qquad
\epsilon\sim\mathcal{N}(0,I).
\]

At inference, \(z\) is sampled from the conditional prior. The sampling
temperature multiplies the prior standard deviation.

### Slot-query decoder

The concatenated curve and latent representation is mapped through a
512-dimensional context MLP. The context is combined with learned slot queries
of width 192. A shared slot decoder produces ten pore descriptors and one
presence logit per slot. A separate head predicts the five global descriptors.
Only slots selected from the predicted presence probabilities are exported, so
the decoded specimen has variable cardinality.

### Generator objective

The base objective is

\[
\begin{aligned}
\mathcal{L}_{\mathrm{base}}={}&
1.00\mathcal{L}_{\mathrm{set}}
+0.35\mathcal{L}_{\mathrm{presence}}
+0.25\mathcal{L}_{\mathrm{macro}}
+1.25\mathcal{L}_{\mathrm{porosity}}\\
&+0.75\mathcal{L}_{\mathrm{spatial}}
+0.20\mathcal{L}_{\mathrm{marginal}}
+0.01\,\alpha_{\mathrm{KL}}\mathcal{L}_{\mathrm{KL}}.
\end{aligned}
\]

`L_set` is a symmetric nearest-neighbour distance between the selected decoded
slots and the target set after feature weighting. `L_presence` is class-balanced
binary cross-entropy. `L_macro`, `L_porosity`, and `L_spatial` are Smooth L1
terms; `L_marginal` compares sorted one-dimensional feature marginals. The KL
term is the diagonal-Gaussian divergence
\(D_{KL}(q_\psi\|p_\phi)\), with 0.02 nat free bits per latent dimension and a
linear warm-up factor \(\alpha_{KL}\).

The frozen PredictNet regularizer adds response consistency,

\[
\mathcal{L}_{\mathrm{train}}=\mathcal{L}_{\mathrm{base}}
+\lambda_r(t)\,\operatorname{SmoothL1}
\left(F_\theta(\hat S,\hat g),y\right),
\]

where the configured terminal weight is \(\lambda_r=0.75\). PredictNet remains
frozen; gradients pass through its input transformation to the generator.

## 3. Forward response predictor

Implementation: `external_predictor/model.py` and
`external_predictor/train_single.py`.

PredictNet receives a padded local tensor with nine channels and three global
descriptors. Coordinates and the remaining pore features are processed by
pointwise Conv1d branches ending at 256 channels. The global branch maps
\(3\rightarrow16\rightarrow32\rightarrow64\rightarrow128\). Cross-attention
uses

\[
Q=XW_Q,\quad K=GW_K,\quad V=GW_V,
\]

\[
\operatorname{Attn}(X,G)=
\operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{256}}\right)VW_O.
\]

A residual connection and maximum pooling over pore rows produce a
512-dimensional specimen embedding. The regression head is
\(512\rightarrow256\rightarrow128\rightarrow64\rightarrow51\).

For normalized outputs \(\hat y\) and labels \(y\), the training objective is

\[
\mathcal{L}_{F}=\operatorname{WMSE}(\hat y,y)
+\lambda_m\operatorname{WMean}[\max(0,\hat y_t-\hat y_{t+1})]
+\lambda_1\operatorname{WMAE}(\hat y,y)
+\lambda_s\operatorname{WMSE}(\Delta^2\hat y,0)
+\lambda_T\operatorname{MSE}_{T}(\hat y,y).
\]

The base and phase-terminal checkpoints have the same architecture but different
label transforms and training objectives. The phase-terminal run uses global
log1p standardization, phase weights `(1, 1, 2)`, six terminal points,
\(\lambda_m=0.5\), \(\lambda_1=0.5\), \(\lambda_s=0.05\), and
\(\lambda_T=1.0\). Checkpoint identity must be reported with every result.

## 4. Evaluation definitions

For a target curve \(y\) and prediction \(\hat y\) on the same grid,

\[
\operatorname{NRMSE}(\hat y,y)=
\frac{\sqrt{T^{-1}\sum_{t=1}^{T}(\hat y_t-y_t)^2}}
{\max_t y_t-\min_t y_t+\varepsilon}.
\]

For \(K\) candidates generated for target \(i\),

\[
e_i^{\mathrm{best@K}}=\min_{1\leq k\leq K}
\operatorname{NRMSE}(F_\theta(S_{ik}),y_i).
\]

Threshold coverage at tolerance \(\tau\) is

\[
C_\tau=\frac{1}{M}\sum_{i=1}^{M}
\mathbf{1}[e_i^{\mathrm{best@K}}\leq\tau].
\]

Coverage from a learned surrogate is not an FE or experimental success rate.

For two standardized structural descriptor vectors \(d_a,d_b\), the supplied
diversity audit uses root-mean-square Euclidean distance,

\[
D_{ab}=\sqrt{P^{-1}\sum_{j=1}^{P}(d_{a,j}-d_{b,j})^2}.
\]

Candidate descriptors are centered by the training median and divided by the
training interquartile range. Nearest-training distance uses the same transform
and metric.

## 5. Reconstruction boundary

`src.optimize_geometry` performs deterministic center redistribution and
porosity matching. `src.export_for_reconstruction` writes the pore parameters
needed by the separate voxel/STL reconstruction stage.

The minimal reviewer repository stops at the verified parameter-export boundary;
the archived voxel and STL utilities do not constitute an Abaqus meshing or
analysis workflow and are therefore excluded. The included stages do not assign
an aluminium constitutive law, generate the final Abaqus mesh, prescribe platen
contact, or define solver steps and boundary conditions. Those FE settings must
be reported from the actual analysis deck rather than inferred from this
repository.
