# Development vs Test-Set Declaration

To guard against benchmark overfitting, every dataset's role in package development is declared here, and the
final test sets + code/config freeze are tagged **before** any full multi-subject array. Reviewers will ask
whether benchmark data influenced algorithm development, default parameters, component-selection rules, plots,
QA metrics, or debugging — this document answers that explicitly.

> Fill the `?` fields during P1–P2. The protocol must be frozen (git tag / preregistration) before P4 full runs.

| Dataset | Used in package dev? | Used for debugging? | Used for parameter/default tuning? | Final locked test subjects/runs | Notes |
|---|---|---|---|---|---|
| ds003620 | ? | ? | ? | ? | tutorials/examples reference this dataset — disclose |
| ds004505 | ? | ? | ? | ? | muscle contrast frozen after feasibility (pre-comparison) |
| ds000117 | ? | ? | ? | ? | raw-vs-SSS branches; 19 subjects |
| ERP CORE N170 | ? | ? | ? | ? | ROI/window adopted from the ERP CORE resource, not chosen post-hoc |
| EEGdenoiseNet | ? | ? | ? | ? | source-level split; no same-recording leakage |
| ds004784 (phantom) | ? | ? | ? | ? | author-created; held-out tuning |

## Freeze record
- Code + config freeze date: `pending`
- Freeze git tag: `pending` (e.g. `benchmark-protocol-v1`)
- Preregistration link (optional): `pending`
- Any change made **after** inspecting locked test results: must be logged here with justification. Default: none.

## Holdout policy
Hold out complete subjects / complete runs / or an entire external dataset (not trials within a subject). Do not
modify default parameters after observing locked test outcomes. Subject exclusion is decided from raw-data QC
criteria evaluated **before** method comparison, blind to method performance.
