# Generative Models for Implied Volatility Surface Path Generation

This repository contains the implementation and empirical experiments developed for a master's thesis on **generative modeling of implied-volatility surface dynamics**, i.e. *"Schrödinger Bridges for Implied Volatility Surface Path Generation"*. The study focuses on the generation of **short-horizon trajectories of complete implied-volatility surfaces**, rather than independent surface samples as explored in [[1]](#ref1).

Two main generative approaches are investigated: **Schrödinger-Bridge-based time-series models** and a **Temporal Latent Flow Matching specification** designed for volatility-surface path generation. The latter extends Flow Matching to the dynamic setting considered here by learning the distribution of the next latent surface increment conditional on a finite history of latent surfaces and recursively generating multi-day trajectories.

The empirical comparison therefore includes three learned temporal specifications:

1. **SBJTS-PCA**, combining a PCA surface representation with a Schrödinger Bridge with Jumps for Time Series[[2]](#ref2);
2. **SBJTS-VAE**, combining the same temporal construction with a nonlinear variational latent representation [[2]](#ref2);
3. **Temporal Latent Flow Matching**, using history-conditioned latent increments and trigonometric interpolation.

Two additional models are retained as fixed references:

4. **LightSB-PCA**, providing a lightweight continuous Schrödinger-Bridge benchmark [[3]](#ref3);
5. **Cont–Vuletic**, providing an arbitrage-aware factor-model and Weighted-Monte-Carlo benchmark [[4]](#ref4).

The models are evaluated on their ability to reproduce both the **cross-sectional geometry** and the **short-horizon dynamics** of SPX implied-volatility surfaces. The comparison considers:

* cross-sectional surface fidelity;
* temporal dependence and short-horizon dynamics;
* marginal and joint distributional similarity to held-out historical paths;
* static-arbitrage consistency.

For the three learned temporal specifications, the experiments additionally study how the amount of historical information used to condition the dynamics affects generated path quality.

---

## Real volatility surfaces as a referential

Real volatility surfaces are used to compare **generated path distributions**, rather than point forecasts of one realized future path.

The setting is the following:

- SPX implied-volatility surfaces on a common **16 maturity x 32 moneyness** grid;
- a chronological train/test split;
- generated paths of **5 trading days**;
- **100 generated paths** per model/configuration in the thesis lag-ablation experiment;
- empirical held-out targets built from rolling 5-day windows of the chronological test sample.

The temporal-conditioning tested are $L \in \{1, 2, 3, 5, 10\}.$

---

## Repository structure
Implied Volatility Surface Path Generation Models
```text
ivs-path-generation/
│
├── src/
│   ├── constructor_surface_cube.py
│   ├── benchmark_utils.py
│   ├── temporal_benchmark_runner.py
│   │
│   ├── volsurface_latentFM/
│   │   ├── __init__.py
│   │   ├── ivs_autoencoder.py
│   │   └── temporal_fm_trigo.py
│   │
│   ├── volsurface_latentSB/
│   │   ├── __init__.py
│   │   ├── sbjts_pca_vol_surface.py
│   │   ├── sbjts_autoencoder_vol_surface.py
│   │   └── lightsb_vol_surface_github_torchcompat.py
│   │
│   └── other_models/
│       ├── __init__.py
│       └── cont_simulations.py
│
├── data/
│   ├── raw/
│   │   └── SPX_Option_Data/
│   └── treated/
│       ├── vol_surface_spx_long.npz
│       ├── benchmark_train.npz
│       ├── benchmark_test.npz
│       └── illustration/
│
├── Illustration.ipynb
├── Paper_illustration.ipynb
├── pyproject.toml
└── README.md
```

---

## Data

The experiments use historical **SPX option-chain data from OptionsDX**.

Raw OptionsDX files are not redistributed in this repository.

```text
data/raw/SPX_Option_Data/
```

The surface-construction code produces a common implied-volatility cube with canonical storage orientation

```text
(moneyness, maturity, date).
```

For model evaluation the surfaces are re-oriented to

```text
(date, maturity, moneyness)
```

and generated paths use

```text
(n_paths, path_length, maturity, moneyness).
```

---

## Build the SPX volatility-surface cube

From the repository root:

```bash
PYTHONPATH=src python -m constructor_surface_cube \
  --input-format spx-txt \
  --data-dir data/raw/SPX_Option_Data \
  --out data/treated/vol_surface_spx_long.npz \
  --verbose
```

Our illustration notebook then works with a chronological training/test split, typically stored as

```text
data/treated/benchmark_train.npz
data/treated/benchmark_test.npz
```

---

## Models

All learned models operate on lower-dimensional representations of the implied-volatility surface. If \(X_t\) denotes the surface observed at date t, its latent representation is denoted by
\[
Z_t \in \mathbb{R}^d.
\]
The models differ primarily in **how the latent dynamics are represented and learned**. SBJTS models the latent trajectory through a controlled jump-diffusion, whereas Temporal Flow Matching learns the conditional distribution of the next latent increment. LightSB and Cont–Vuletić are kept as fixed comparison benchmarks.

---

### 1. SBJTS-PCA

Implementation:
```text
src/volsurface_latentSB/sbjts_pca_vol_surface.py
```
SBJTS-PCA combines a linear PCA representation of the volatility surfacewith the Schrödinger Bridges with Jumps for Time Series (SBJTS)construction.
Historical surfaces are first projected onto a d-dimensional PCA representation,
$$X_t \rightarrow Z_t \in R^d$$
The resulting chronological sequence $Z_{t_1},…,Z_{t_N}$ is treated as the observed latent time series. SBJTS constructs acontrolled jump-diffusion whose finite-dimensional law is fitted to thejoint distribution of the observed latent trajectory. The continuous component captures gradual latent movements, while the jump componentallows discontinuous changes in the surface dynamics. Temporal dependence is controlled through memory_order=L with $L \in \{1, 2, 3, 5, 10\}$.
Thus,  represents the amount of recent latent history used whenestimating the next transition; it is distinct from the length of thegenerated path. After simulation, generated latent states are mapped back through the PCA reconstruction to obtain full implied-volatility-surface trajectories.

---

### 2. SBJTS-VAE

Implementation:

```text
src/volsurface_latentSB/sbjts_autoencoder_vol_surface.py
```

SBJTS-VAE retains the same SBJTS temporal construction but replaces the linear PCA representation with a learned nonlinear variational latent space. The VAE associates each surface with a conditional latent distribution
$$    q_\phi(z\mid X)
    =
    \mathcal{N}
    \left(
        \mu_\phi(X),
        \operatorname{diag}\left(\sigma_\phi^2(X)\right)
    \right)$$

For the subsequent time-series model, the posterior mean is used as the
deterministic latent representation,
$$Z_t = \mu_\phi(X_t)$$

This produces the chronological latent trajectory on which the SBJTS jump-diffusion is fitted. Generated latent paths are subsequently reconstructed through the learned decoder.

The purpose of this specification is to test whether a nonlinear, regularized latent representation improves path generation relative to the linear PCA representation while keeping the underlying SBJTS dynamics comparable.

As for SBJTS-PCA, temporal conditioning is varied through
$$    \texttt{memory\_order} = L,
    \qquad
    L \in \{2,3,5,10\}$$

---

### 3. Temporal Latent Flow Matching

Implementation:

```text
src/volsurface_latentFM/ivs_autoencoder.py
src/volsurface_latentFM/temporal_fm_trigo.py
```

The Flow-Matching specification is formulated in a learned latent space. Each historical implied-volatility surface \(X_t\) is first mapped by the encoder \(E\) to a lower-dimensional latent representation

$$
Z_t = E(X_t) \in \mathbb{R}^d.
$$

Rather than modeling the surfaces independently, Temporal Flow Matching models the next latent increment

$$
\Delta Z_t = Z_{t+1} - Z_t
$$

conditional on the recent history of latent representations,

$$
C_t =
\left(
Z_{t-L+1},\ldots,Z_t
\right).
$$

For the artificial Flow-Matching time \(s\in[0,1]\), the final specification uses the trigonometric interpolation

$$
X_s
=
\cos\left(\frac{\pi s}{2}\right)\varepsilon
+
\sin\left(\frac{\pi s}{2}\right)\Delta Z_t,
$$

where \(\varepsilon\) is sampled from the reference noise distribution. The corresponding target velocity is

$$
u_s
=
-\frac{\pi}{2}
\sin\left(\frac{\pi s}{2}\right)\varepsilon
+
\frac{\pi}{2}
\cos\left(\frac{\pi s}{2}\right)\Delta Z_t.
$$

The conditional velocity field

$$
v_\theta(X_s,s\mid C_t)
$$

is trained to transport the reference noise toward the distribution of latent increments conditional on the recent history.

At generation time, the resulting latent increment is applied recursively,

$$
Z_{t+1}
=
Z_t+\widehat{\Delta Z}_t,
$$

and the generated state \(Z_{t+1}\) is added to the conditioning history for the next transition. Repeating this procedure produces a complete multi-day latent trajectory

$$
(Z_{t+1},Z_{t+2},\ldots,Z_{t+H}),
$$

which is subsequently reconstructed into a path of implied-volatility surfaces using the learned decoder.

Temporal conditioning is controlled through

```python
TemporalFMConfig(context_lags=L)
```

with $L \in \{1, 2, 3, 5, 10\}$.
Here, \(t\) denotes chronological market time, whereas \(s\) is the artificial Flow-Matching interpolation time used to generate each latent transition.

---

### 4. LightSB

Implementation:

```text
src/volsurface_latentSB/lightsb_vol_surface_github_torchcompat.py
```

LightSB is used as a lightweight continuous Schrödinger-bridge benchmark. In our implementation, it is applied in PCA latent space: each implied-volatility surface \(X_t\) is first projected onto a low-dimensional representation,

$$
X_t \longmapsto Z_t \in \mathbb{R}^d.
$$

LightSB solves a classical Schrödinger Bridge with a Wiener reference process. Given two prescribed latent distributions \(p_0\) and \(p_1\), the bridge seeks a path measure \(P^\star\) that remains close, in relative entropy, to the reference process while satisfying the endpoint
constraints,

$$
P^\star
\in
\arg\min_{\substack{P\\P_0=p_0,\;P_1=p_1}}
H(P\mid Q).
$$

Rather than learning the full path measure directly, LightSB exploits the equivalence between the dynamic Schrödinger Bridge and its static entropic optimal-transport formulation. It learns an approximation of the optimal endpoint coupling,

$$
\pi_\theta(Z_0,Z_1)
\approx
\pi^\star(Z_0,Z_1),
$$

and intermediate latent states are generated according to the corresponding Brownian bridge dynamics. The resulting latent trajectories are then reconstructed into implied-volatility-surface paths using the PCA representation.

Compared with SBJTS, LightSB therefore provides a simpler **continuous-diffusion Schrödinger-bridge reference**: it uses endpoint marginal constraints and a Wiener reference process, whereas SBJTS models the finite-dimensional joint law of the latent time series using a jump-diffusion construction.

LightSB is a **fixed reference model** in the conditioning-lag ablation. It is not assigned a conditioning order \(L\); the same LightSB result is shown across lag dashboards solely to provide a common benchmark.

---

### 5. Cont–Vuletić Arbitrage-Aware Reference

Implementation:

```text
src/other_models/cont_simulations.py
```

The Cont–Vuletić benchmark follows the arbitrage-aware scenario-generation framework of Cont and Vuletić. Each implied-volatility surface \(X_t\) is first represented through a low-dimensional factor representation,

$$
X_t \longmapsto Z_t \in \mathbb{R}^d,
$$

where \(Z_t\) contains the factors used to describe the main variations of
the volatility surface.

The temporal evolution of these factors is then modeled statistically to generate candidate latent trajectories,

$$
(Z_{t_1},\ldots,Z_{t_H}),
$$

which are reconstructed into candidate implied-volatility-surface paths,

$$
(X_{t_1},\ldots,X_{t_H}).
$$

The distinctive feature of the Cont–Vuletić approach is that static arbitrage is imposed at the **path-selection level**. For each candidate trajectory \(\omega_i\), a cumulative penalty

$$
\phi(\omega_i)
$$

measures the static-arbitrage violations encountered along the generated surface path. Candidate trajectories are then reweighted using a Weighted Monte Carlo procedure,

$$
w_i(\beta)
=
\frac{
    \exp\left[-\beta\,\phi(\omega_i)\right]
}{
    \sum_j
    \exp\left[-\beta\,\phi(\omega_j)\right]
},
$$

where \(\beta\geq0\) controls the strength of the arbitrage penalization. Paths with larger violations therefore receive smaller sampling weights, while paths with lower arbitrage penalties are favored.

The benchmark thus combines two components:

1. a low-dimensional statistical model for the temporal evolution of the implied-volatility surface; and
2. an explicit path-level reweighting mechanism designed to favor arbitrage-consistent trajectories.

This makes the construction conceptually different from SBJTS and Temporal Flow Matching: the underlying dynamics first generate candidate paths,
while financial consistency is subsequently introduced through the Weighted Monte Carlo distribution over those paths.

Like LightSB, Cont–Vuletić is kept fixed in the conditioning-lag experiment and is not assigned an artificial conditioning order \(L\).

---

## Results

The main reproducibility notebook is

```text
Illustration
```

Core settings are

```text
conditioning lags : 1, 2, 3, 5, 10
generated paths   : 100
forecast horizon  : 5 trading days
candidate paths   : 500
random seed       : 42
```

The **conditioning lag** and **generation horizon** are distinct quantities. For example, \(L=10\) means that the model uses ten historical latent observations to condition each transition; it does not mean that the generated path must contain ten future dates.

---

## Evaluation

The held-out empirical distribution is formed from rolling 5-day windows of the chronological test sample.

The comparison deliberately does not pair a generated path with one specific realized future path. Instead, the objective is to assess whether the models reproduce the statistical distribution and dynamics of realistic volatility-surface paths.

The final diagnostics include:

#### Cross-sectional surface structure

- mean ATM term structure;
- mean smile;
- downside and upside skew;
- wing-spread / smile-shape proxies;
- mean generated volatility surface;
- absolute mean-surface error.

#### Dynamic path structure

- daily implied-volatility increment distribution;
- full-surface step magnitude;
- distribution of short-horizon surface movements.

#### Distributional distance

Distributional fidelity is evaluated using **Wasserstein-2 ($W_2$) distances** at three complementary levels.

**Marginal diagnostic $W_2$:** For scalar diagnostics, the corresponding observations are pooled separately across the real and generated samples. If

$$
\widehat P=\frac{1}{N_r}\sum_{i=1}^{N_r}\delta_{x_i^{\mathrm{real}}},
\qquad
\widehat Q=\frac{1}{N_g}\sum_{j=1}^{N_g}\delta_{x_j^{\mathrm{gen}}},
$$

the empirical one-dimensional Wasserstein-2 distance is

$$
W_2(\widehat P,\widehat Q)
=
\left(
\int_0^1
\left|
\widehat F_{\mathrm{real}}^{-1}(u)
-
\widehat F_{\mathrm{gen}}^{-1}(u)
\right|^2du
\right)^{1/2}.
$$

This is computed for pooled **IV levels, daily IV increments, surface-step magnitudes, ATM IV, skew, and wing-spread/smile-shape proxies**. These metrics compare marginal feature distributions and do not preserve dependence across the IV grid or through time. In particular, we can also see it also as “Surface $W_2$” in commun plots, and **IV-level $W_2$** in the lastest.

**Surface OT $W_2$:** To assess the joint distribution of complete surfaces, each $16\times32$ IV surface is vectorized as

$$
S=\operatorname{vec}(\sigma)\in\mathbb R^{512}.
$$

The distance between one real surface $S_i^{\mathrm{real}}$ and one generated surface $S_j^{\mathrm{gen}}$ is measured by their Euclidean distance in this $512$-dimensional space, i.e its squared value is the sum of the squared IV differences over all maturity-moneyness grid points,

$$
\left\|
S_i^{\mathrm{real}}-S_j^{\mathrm{gen}}
\right\|_2^2
=
\sum_{k=1}^{512}
\left(
S_{i,k}^{\mathrm{real}}
-
S_{j,k}^{\mathrm{gen}}
\right)^2.
$$

The Wasserstein distance does not compare surfaces one by one using an arbitrary pairing. Instead, it finds the optimal way of matching probability mass between the empirical real and generated surface distributions. If $\pi_{ij}$ denotes the amount of probability mass transported from real surface $i$ to generated surface $j$, then

$$
W_{2,\mathrm{surf}}^2
=
\min_{\pi\in\Pi(\widehat\mu_{\mathrm{real}},\widehat\mu_{\mathrm{gen}})}
\sum_{i,j}
\pi_{ij}
\left\|
S_i^{\mathrm{real}}
-
S_j^{\mathrm{gen}}
\right\|_2^2.
$$

Here, $\Pi(\widehat\mu_{\mathrm{real}},\widehat\mu_{\mathrm{gen}})$ is the set of all admissible matchings whose marginals are the empirical real and generated distributions. The optimization therefore selects the matching that minimizes the average squared discrepancy between complete surfaces.

Unlike IV-level $W_2$, which pools individual IV values, **Surface OT $W_2$ treats each entire $16\times32$ surface as one observation**. It therefore preserves the joint cross-sectional structure of the volatility surface when comparing the real and generated distributions.


**PCA path OT $W_2$:** Temporal distributional fidelity is evaluated on complete five-day paths in a common PCA representation. A PCA basis with $d=8$ components is fitted on the real surfaces and applied unchanged to both real and generated data,

$$
Z_t
=
U_d^\top
\left(
\operatorname{vec}(S_t)-\bar S
\right)
\in\mathbb R^8.
$$

Each five-day path is represented by the concatenated vector

$$
Y_t
=
\operatorname{vec}(Z_{t+1},\ldots,Z_{t+5})
\in\mathbb R^{40},
$$

and $W_{2,\mathrm{path}}$ is computed between the empirical distributions of these path vectors. This provides a joint measure of **cross-sectional surface structure and temporal dependence across the full five-day horizon**, while avoiding optimal transport directly in the $5\times16\times32=2560$-dimensional raw path space.

The three measures are reported separately: **IV-level $W_2$** evaluates marginal IV distributions, **Surface OT $W_2$** evaluates complete surface distributions, and **PCA path OT $W_2$** evaluates complete five-day trajectory distributions. They are not averaged because they are defined in spaces of different dimensions and numerical scales.


#### Static-arbitrage diagnostics

Generated surfaces are also checked for violations of

- calendar monotonicity in total variance;
- call-price monotonicity in strike/moneyness;
- butterfly convexity of call prices.

The wing-spread quantity used in the distributional diagnostics is only a smile-shape proxy and should not be confused with mathematical call-price convexity.

---

## Running the benchmark

Create and activate a virtual environment and install the package in editable mode:

```bash
cd /path/to/shrodinger_bridge_models

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e .
```

Then launch Jupyter and run

```text
Illustration.ipynb
```

from the repository root.

The notebook imports the final model implementations directly and saves/resumes generated path files so that completed configurations do not need to be retrained.


## Reproducibility note

The final results should be reproduced from the chronological train/test split rather than from randomly shuffled surfaces.

Random seeds are fixed where supported, but neural-network training can still exhibit small hardware- or backend-dependent numerical differences.

## References

<a id="ref1"></a>
[1] Y. Liu, I. Ben Tahar, O. Brooks and D. Bajalica, *Latent Flow Matching for Arbitrage-Aware Implied Volatility Surface Generation*, preprint, (2026).

<a id="ref2"></a>
[2] S. De Marco, H. Pham and D. Zanni, *Schrödinger bridges with jumps for time series generation*, preprint, (2026).

<a id="ref3"></a>
[3] A. Korotin, N. Gushchin, and E. Burnaev, *Light Schrödinger Bridge*, International Conference on Learning Representation, (2024).

<a id="ref4"></a>
[4] M. Vuletic and R. Cont, *Simulation of arbitrage-free implied volatility surfaces*, SSRN, (2022).



