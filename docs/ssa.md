# Singular Spectrum Analysis

Singular Spectrum Analysis (SSA) represents a scalar time series in a
delay-coordinate space and separates structures according to the geometry of
that space. The module provides two scientifically distinct methods:

- **Basic SSA** produces an additive decomposition into reconstructed
  components. An optional frequency-based grouping rule selects components for
  subtraction.
- **Local SSA** follows Teixeira et al.[^2] It models a high-amplitude artifact
  through locally estimated subspaces of clustered delay vectors and subtracts
  the reconstructed artifact.

Both methods are univariate. For multichannel data, each channel is processed
independently; there is no joint embedding or cross-channel covariance model.
This distinction matters scientifically: independent channel-wise SSA is not
multivariate SSA (MSSA).

## Delay-coordinate representation

Let $x_1,\ldots,x_N$ be a time series and choose an embedding dimension $L$.
With $K=N-L+1$, the trajectory matrix is

$$
\mathbf{X}=
\begin{bmatrix}
x_1 & x_2 & \cdots & x_K \\
x_2 & x_3 & \cdots & x_{K+1} \\
\vdots & \vdots & \ddots & \vdots \\
x_L & x_{L+1} & \cdots & x_N
\end{bmatrix}.
$$

Every column is an overlapping length-$L$ delay vector. Because equal elements
of the original series lie on the anti-diagonals, $\mathbf{X}$ is Hankel.
Temporal regularity becomes low-dimensional structure in this embedded space.

## Basic SSA decomposition

Basic SSA factorizes the trajectory matrix as

$$
\mathbf{X}=\sum_{i=1}^{d}
\sigma_i\mathbf{u}_i\mathbf{v}_i^{\mathsf T},
$$

where $\sigma_1\geq\cdots\geq\sigma_d\geq0$. Each singular triplet defines an
elementary trajectory matrix. Direct SVD is algebraically equivalent to
eigendecomposing a lag-covariance matrix, while avoiding the loss of numerical
conditioning caused by forming a matrix product.[^1]

Each elementary matrix is converted back to a time series by averaging its
anti-diagonals. The number of matrix entries contributing to a reconstructed
sample increases near the center of the record and decreases at its edges.
The complete set of reconstructed components therefore satisfies

$$
x_t=\sum_{i=1}^{d}\widetilde{x}^{(i)}_t
$$

up to floating-point precision.

```python
from mne_denoise.ssa import ssa_decompose, ssa_w_correlation

components, diagnostics = ssa_decompose(signal, window_length=100)
w_correlation = ssa_w_correlation(components, diagnostics["window_length"])
```

### Component separability

W-correlation quantifies the weighted correlation between reconstructed
components. The weights are the anti-diagonal multiplicities used during
reconstruction. Values near zero indicate that two reconstructed components
are well separated; large magnitudes indicate mixing. W-correlation is a
diagnostic of the chosen embedding and decomposition, not an automatic
criterion for identifying an artifact.

## Frequency-based component grouping

The frequency-guided cleaner estimates the dominant frequency $f_i$ of every
reconstructed component using an $N$-point real Fourier transform. It forms an
artifact index set

$$
\mathcal{A}=\{i:f_i\leq f_{\max}\}
$$

for low-frequency removal, or

$$
\mathcal{A}=\{i:f_{\min}\leq f_i\leq f_{\max}\}
$$

for a specified rejection band. The cleaned series is

$$
x_t^{\mathrm{clean}}=x_t-
\sum_{i\in\mathcal{A}}\widetilde{x}^{(i)}_t.
$$

This dominant-frequency rule is a grouping strategy supplied by
`mne-denoise`; it is not part of the definition of Basic SSA. The scientific
interpretation depends on several properties:

- the Fourier-bin spacing is $f_s/N$, so a decision near a boundary can change
  with record length;
- DC is a possible dominant frequency;
- a broadband component is classified by its largest spectral bin, not by its
  total power inside the rejection band;
- restricting `n_check` to the leading components can miss a lower-variance
  artifact, so the default examines all numerical-rank components; and
- genuine neural activity whose component peak lies in the rejected range can
  also be removed.

The selected frequencies, component indices, singular values, and
W-correlation structure should therefore be examined when the result is used
for scientific inference.

```python
from mne_denoise.ssa import compute_basic_ssa

cleaned, diagnostics = compute_basic_ssa(
    data,
    sfreq=250.0,
    window_seconds=0.5,
    drop_freq_max=3.0,
)
print(diagnostics["dropped_frequencies"])
```

## Local SSA for high-amplitude artifacts

Local SSA is based on a different signal model. Rather than grouping the
global Basic SSA eigentriples, it assumes that a high-amplitude artifact forms
locally coherent structure among delay vectors.[^2] Its stages are:

1. Embed the observed series into overlapping delay vectors.
2. Partition the delay vectors into $q$ clusters using k-means.
3. Center the vectors separately within each cluster.
4. Eigendecompose each local covariance matrix.
5. Estimate a separate signal-subspace dimension in every cluster using the
   minimum-description-length criterion.
6. Project each cluster onto its selected local subspace and restore its mean.
7. Return the reconstructed vectors to their original temporal positions and
   apply anti-diagonal averaging.
8. Interpret the locally coherent reconstruction as the artifact and subtract
   it from the observation.

For a cluster with $N_c$ vectors of dimension $M$, let
$\lambda_1\geq\cdots\geq\lambda_M$ denote the covariance eigenvalues. For each
candidate subspace order $k$, let $G_k$ and $A_k$ be the geometric and
arithmetic means of the discarded eigenvalues. The selected order minimizes

$$
\operatorname{MDL}(k)=
-N_c(M-k)\log\left(\frac{G_k}{A_k}\right)
+\frac{1}{2}K(k)\log N_c,
$$

where

$$
K(k)=kM-\frac{k(k-1)}{2}+1.
$$

```python
from mne_denoise.ssa import compute_local_ssa

cleaned, diagnostics = compute_local_ssa(
    data,
    window_length=41,
    n_clusters=6,
    random_state=0,
)
print(diagnostics["n_clusters"])
print(diagnostics["subspace_dimensions"])
```

### Assumptions and interpretation

The local reconstruction is treated as artifact because the method assumes
that coherent, high-energy activity occupies the leading local subspaces and
that the desired EEG behaves more like a residual process. This is a modeling
assumption, not a universal property of EEG. High-amplitude rhythmic neural
activity can also occupy those subspaces and be attenuated.

Teixeira et al. report changes mainly at low frequencies and relative
preservation of beta-band activity in their experiments.[^2] Those observations
do not establish frequency-specific guarantees for other recordings,
montages, populations, or artifact types.

### Underspecified numerical choices

The source does not uniquely specify k-means initialization and restart count,
a universal maximum value of $q$, random seeding, zero-eigenvalue handling, or
the exact fallback when a clustering is too small for reliable local covariance
estimation. The implementation makes these choices explicit:

- a fixed `random_state` makes clustering reproducible;
- every cluster must contain at least `window_length` delay vectors;
- `n_clusters="auto"` searches downward from
  `min(max_clusters, n_vectors // window_length)` until the reliability
  conditions are satisfied;
- `max_clusters=10` is a computational default, not a value prescribed by the
  paper; and
- logarithms in the MDL calculation are stabilized for zero eigenvalues.

Cluster labels, eigenvalues, MDL scores, and selected subspace dimensions are
available as diagnostics so these numerical decisions can be inspected.

## Choosing the embedding dimension

For Basic SSA, the canonical orientation satisfies
$2\leq L\leq K=N-L+1$, including $L=K$ for odd $N$. Larger $L$ provides a
higher-dimensional representation and finer potential separation, but
increases computation and leaves fewer delay vectors for estimation.

In the absence of prior information, Basic SSA references often use
$L\approx N/2$; when a periodicity is known, $L$ can instead be related to its
period.[^1] For local SSA, Teixeira et al. state $M>f_s/f_r$ as a lower bound
associated with resolving a frequency $f_r$, and report that larger embeddings
improved separation at low signal-to-noise ratios.[^2]

The software caps automatically selected windows through `max_window` to limit
the cost of dense decompositions. That cap is a computational safeguard, not a
scientific recommendation. Window choice should be reported as part of an
analysis and assessed with component and reconstruction diagnostics.

## Record boundaries and estimator semantics

SSA is record-dependent. Changing $N$ changes the trajectory matrix, and can
also change Fourier bins, cluster membership, and estimated local subspaces.
The estimators are therefore transductive:

- `fit` validates parameters and records the channel layout;
- `transform` decomposes the records supplied to that call;
- no decomposition learned from one record is reused for another; and
- epochs are processed independently.

Consequently, decomposing a continuous recording and decomposing separately
segmented epochs are different scientific operations. Record boundaries should
be chosen before analysis and reported alongside the SSA parameters.

## Scientific scope and limitations

- The methods do not test whether a component is statistically significant;
  Monte Carlo SSA would require a specified stochastic null model, surrogate
  generation, significance levels, and multiple-testing control.
- The methods require finite observations; missing-data SSA would require an
  explicit observation mask, iterative reconstruction model, convergence
  criterion, and rules for distinguishing observed from imputed EEG.
- The methods reconstruct records at their original length; SSA forecasting
  requires separate recurrent or vector forecasting assumptions.
- Dense decompositions provide the numerical reference behavior. Truncated or
  fast Hankel solvers require explicit rank-selection semantics and parity
  validation, while local MDL additionally depends on the discarded
  eigenvalue spectrum.
- Exact reproduction of the Teixeira et al. figures is not possible because
  the EEG recordings, noise realizations, and complete synthetic waveform
  definitions are not publicly available. Validation therefore targets the
  published equations, reconstruction identities, documented parameter grid,
  and controlled synthetic signals.

## References

[^1]: N. Golyandina and A. Zhigljavsky, *Singular Spectrum Analysis for Time
    Series*, Springer, 2013. <https://doi.org/10.1007/978-3-642-34913-3>
[^2]: A. R. Teixeira et al., "Automatic removal of high-amplitude artefacts
    from single-channel electroencephalograms," *Computer Methods and Programs
    in Biomedicine*, 83, 125–138, 2006.
    <https://doi.org/10.1016/j.cmpb.2006.06.003>
