# Sensor Noise Suppression (SNS)

SNS suppresses sensor-specific noise by reconstructing each sensor from
spatially redundant neighboring sensors. It assumes that signals of interest
are represented across multiple sensors, whereas the targeted sensor noise is
specific to individual sensors {footcite:p}`decheveigne_simon2008_sensor`.

## Usage

```python
from mne_denoise.sns import SNS

model = SNS(n_neighbors=4, n_iter=1)
clean = model.fit_transform(data)  # data: (n_channels, n_times)
```

## Key points

- n_neighbors, skip, and the neighbor-selection rule determine the local
  spatial predictors.
- sample_weight, robust masking, and n_iter affect covariance estimation and
  repeated passes; inspect the fitted ranks and rejected-sample count.
- NumPy input is channel-first. The estimator preserves supported MNE
  containers and uses the fitted channel layout for later transforms.
- compute_sns_weights exposes the local weights; compute_sns provides the
  one-shot array operation.

SNS is not a generic spatial filter: it relies on sensor-specific noise and
spatial redundancy being reasonable for the recording.

## References

```{footbibliography}
```
