# Sensor Noise Suppression (SNS)

The `mne_denoise.sns` module implements Sensor Noise Suppression (SNS) [^1], a
spatial method that regenerates each sensor from correlated neighboring
sensors. It targets noise that is specific to individual sensors while
preserving signals that are spatially redundant across the array.

SNS does not target ocular, cardiac, muscle, or other artifacts that are
themselves correlated across sensors. Its assumptions should be checked for the
recording system and scientific signal of interest.

## Estimator API

```python
from mne_denoise.sns import SNS

model = SNS(n_neighbors=20)
model.fit(training_data)
cleaned = model.transform(evaluation_data)
```

`fit()` learns a channel mean and spatial operator from the training data.
`transform()` always uses that fitted mean, so a sample has the same result
whether transformed alone, in a temporal chunk, or among other epochs. By
default the fitted mean is subtracted and is not added back to the output.
Set `preserve_mean=True` to restore the fitted training mean:

```python
cleaned = SNS(n_neighbors=20, preserve_mean=True).fit_transform(data)
```

For MNE Raw, Epochs, and Evoked inputs, SNS automatically selects one
homogeneous data-channel type and returns a copy of the same container. Timing,
annotations, events, epoch metadata, averaging information, and excluded
channels are preserved. Fit and transform must use the same selected channel
names in the same order.

## Robust, iterative, and chunked fitting

Global sample weights prevent selected samples from influencing the fitted mean
or projections. Continuous weights have shape `(n_times,)`; epoched weights have
shape `(n_epochs, n_times)`.

```python
weights = good_samples.astype(float)
model = SNS(n_neighbors=20, n_iter=2, outlier_threshold=8.0, chunk_size=10_000)
cleaned = model.fit_transform(data, sample_weight=weights)
```

`outlier_threshold` applies an additional fixed binary mask while fitting. For
each channel, SNS computes a median and `1.4826 * MAD` scale from manually
included samples, falls back to standard deviation and then unit scale when
needed, and rejects a sample if any channel exceeds the threshold. The combined
manual and automatic weights remain fixed across iterations. The learned
operator is applied to all samples, including rejected fitting samples.

With `n_iter > 1`, SNS learns a new projection from the output of each pass and
composes every pass into one fixed `denoising_matrix_`. `chunk_size` bounds the
temporary arrays used for weighted statistics and operator application. The
shared MNE extractor still materializes MNE input, so this option is not an
out-of-core MNE reader.

## One-shot and covariance APIs

`compute_sns()` self-centers a continuous array, learns the requested passes,
and applies them immediately:

```python
from mne_denoise.sns import compute_sns, compute_sns_weights

cleaned, diagnostics = compute_sns(
    data, n_neighbors=20, n_iter=2, outlier_threshold=8.0
)

centered = data - data.mean(axis=1, keepdims=True)
covariance = centered @ centered.T / centered.shape[1]
operator, n_used, ranks = compute_sns_weights(covariance, n_neighbors=20)
```

`compute_sns_weights()` is the covariance-level, single-pass primitive.
`compute_sns()` is the main array algorithm used by `SNS`; it adds centering,
weighting, masking, iteration, and chunking.

## Diagnostics

After fitting, the estimator exposes:

- `training_mean_`: weighted training mean used by every transform;
- `denoising_matrix_`: composed spatial operator;
- `denoising_matrices_`: operator learned at each iteration;
- `neighbor_ranks_` and `neighbor_ranks_per_iteration_`: numerical local ranks;
- `input_rank_`, `effective_weight_sum_`, and `rejected_sample_count_`;
- `n_neighbors_` and `n_iter_`: effective fitted operating point.

The one-shot function returns the corresponding values in its diagnostics
dictionary. It reports only the effective weight sum and rejection count, not a
full sample-weight vector.

## Choosing parameters

- `n_neighbors=0` uses all eligible sensors. A smaller set is faster and may be
  preferable for dense arrays.
- `skip` omits the most correlated neighbors, which can help when adjacent
  sensors share local noise.
- Start with `n_iter=1`; repeated passes are more aggressive and typically show
  diminishing changes.
- Enable robust masking only after inspecting the data distribution. A threshold
  near 8 is a useful demonstration value, not a universal recommendation.
- Compare signal preservation and sensor-noise attenuation on data representative
  of the intended analysis before adopting SNS in a pipeline.

See the {ref}`sphx_glr_auto_examples_sns` gallery for basic usage and a
deterministic exploration of the algorithm's assumptions and diagnostics. The
standalone [`scripts/replicate_sns_paper.py`](../scripts/replicate_sns_paper.py)
runs the larger synthetic experiments used for paper-oriented reproducibility.

## References

[^1]: de Cheveigné, A., & Simon, J. Z. (2008). Sensor noise suppression.
    *Journal of Neuroscience Methods, 168*(1), 195–202.
    <https://doi.org/10.1016/j.jneumeth.2007.09.012>
