# Machine Learning Tutorial

For developers who want to understand the algorithms in this codebase
from first principles. **No prior ML background assumed**, but you
should be comfortable with high-school-level algebra and willing to
follow some matrix notation. Each section starts with intuition, then
the math, then where it lives in the code.

If you just want to know what to call when, see `DEVELOPER_INTRO.md`
instead. This document is for understanding *why* the algorithms work.

---

## Contents

1. [The big picture: what problems are we solving?](#1-the-big-picture)
2. [Cat identification — CLIP embeddings + cosine similarity](#2-cat-id--clip)
3. [Robust statistics — median and MAD](#3-robust-statistics)
4. [Log-transforms — why we work in log-space for gas](#4-log-transforms)
5. [The gas anomaly detector — log-Gaussian z-scores](#5-gas-anomaly)
6. [PCA / eigendecomposition — finding the shape of waveforms](#6-pca-eigen)
7. [Tikhonov regularization — taming small sample sizes](#7-tikhonov)
8. [Gaussian Mixture Models + BIC — finding clusters](#8-gmm-bic)
9. [The trend detector — Welch-style mean-shift z-test](#9-welch-trend)
10. [Operating characteristics — TPR, FPR, ROC, calibration](#10-oc)
11. [Why each algorithm was picked for its job](#11-why-each)
12. [Further reading](#12-further-reading)

---

## 1. The big picture

The litter box monitor is an **anomaly detection system**. It is
*not* a classifier (no "this cat has kidney disease" labels), it is
*not* a regression (no "this cat has X years left"). It just asks:

> Given everything we know about this cat's normal behaviour,
> does this new observation look usual or unusual?

That single question shows up in five different forms in the code, and
each form gets its own algorithm:

| Question | Algorithm |
|---|---|
| "Whose photo is this?" | CLIP embedding + cosine similarity nearest-neighbour |
| "Is anything visually concerning in this image?" | GPT-4o vision (a large language model with vision capability) |
| "Is the gas level on this visit unusually high for this cat?" | Robust log-Gaussian fit + z-score |
| "Does the weight WAVEFORM look like this cat's typical pattern?" | PCA + GMM clustering |
| "Has this cat been gradually drifting over weeks?" | Welch-style two-sample mean-shift test |

The unifying mathematical idea is **z-scores**: how many standard
deviations is this observation away from what we'd expect under the
"normal" model? If z is small, fine. If z is large, alarm.

The tricky part is building good "normal" models. Naïve approaches
(mean ± standard deviation) break in real world data because:

- Sensors are noisy and produce occasional outliers
- One cat's normal differs wildly from another's
- Slow drift is a thing
- Sample sizes are small for new cats

The choices in this codebase are all aimed at being **robust** in those
ways.

---

## 2. Cat identification — CLIP embeddings + cosine similarity { #2-cat-id--clip }

### The problem

A camera takes a photo of the cat in the box. We have N reference photos
of registered cats. Which cat does the new photo match?

### The naïve approach (and why it fails)

You could do pixel-by-pixel comparison: subtract the new image from
each reference, compute total difference, pick the smallest. This is
terrible — a cat in a slightly different pose, lighting, or framing
will score as wildly different from itself.

### The CLIP idea

CLIP (Contrastive Language-Image Pretraining) is a neural network
trained by OpenAI on hundreds of millions of image-caption pairs. The
key trick: it was trained so that the image of a cat and the caption
"a photo of a cat" both get mapped to nearby points in a 512-dimensional
embedding space. By extension, two photos of the same cat (even at
different angles/lighting) end up close in that space, while two photos
of different cats end up far apart.

We use the open-source `clip-ViT-B-32` model from sentence-transformers.
"ViT-B/32" means a Vision Transformer with patches of 32×32 pixels.

### The math

For each image, CLIP outputs a vector `v ∈ ℝ⁵¹²`. To compare two
vectors, we use **cosine similarity**:

```
                     v₁ · v₂
   cos(v₁, v₂) = ─────────────────
                  ‖v₁‖ × ‖v₂‖
```

This is the cosine of the angle between the two vectors. It's in
[−1, +1]:

- **+1** = perfectly aligned (the model thinks they're the same thing)
- **0** = orthogonal (totally unrelated)
- **−1** = opposite (rare with image embeddings)

For our purposes the values are usually in [0, 1] because CLIP
embeddings tend to be in similar directions for related content.

### How it's used here

- `src/litterbox/embeddings.py` embeds each cat's reference photos at
  registration time and stores the vectors in a Chroma database.
- At identification time, the new visit's photo is embedded and
  Chroma returns the top candidates by cosine similarity.
- The threshold `ID_THRESHOLD = 0.82` is what we consider "confident
  enough to call this a match". Below that, we either fall back to
  GPT-4o for a side-by-side visual check, or mark the visit Unknown.

The threshold was chosen empirically — high enough to avoid confusing
similar-looking cats, low enough that legitimate matches don't get
rejected. See `simulator/run_simulation.py`'s identity accuracy results
for empirical validation (typically 70–90% correct identification in a
4-cat household).

### Failure modes

- Two visually similar cats (same colour, same fur length) can score
  >0.82 against each other → wrong tentative ID, requires manual
  confirmation.
- A cat in unusual conditions (wet fur, missing fur from a recent vet
  visit) may score below 0.82 against its own reference → marked
  Unknown.
- The model was trained on natural photos, not litter-box-cam top-down
  shots. Cropped or partial views perform worse than full body shots.

The two-stage pipeline (CLIP nearest-neighbour, then GPT-4o
side-by-side comparison if close) was added specifically to reduce
false positives from the first failure mode.

---

## 3. Robust statistics — median and MAD { #3-robust-statistics }

### The problem

You've collected 30 readings of "ammonia peak" from Luna's last 30
visits. You want to characterise her normal range so you can flag a
new visit as unusual or not. What summary statistics do you use?

### Why mean + std fails

The textbook answer is `mean ± k × standard deviation`. This has a
fatal flaw for our data: it's not robust to outliers.

Concrete example: suppose Luna's 30 visits have NH₃ readings of
[40, 42, 38, 45, ..., 39] — all in the 35–50 range — except for
one anomalous visit where she read 300 ppb. The mean now becomes
~50 (pulled up by the outlier) and the standard deviation jumps from
~5 to ~50. Now ANY future reading in the 100–200 range looks "within
2σ of normal" — the detector is broken precisely because of the
contamination it should have caught.

This is called **breakdown point** in statistics: how many bad data
points can sneak in before your summary becomes meaningless?

- Mean / std: **0% breakdown point**. One bad value contaminates them.
- Median / MAD: **50% breakdown point**. Up to half the data can be
  garbage and the summary is still meaningful.

For a chronically-monitored cat, prior visits in the historical pool
will include earlier anomalies. We *cannot* let those contaminate the
"normal" model — that would suppress detection of new anomalies of
the same kind.

### The math

**Median** of a sample is the middle value when the sample is sorted.
For [3, 5, 7, 9, 11], the median is 7. Robust to outliers because
moving the smallest or largest value doesn't change the middle.

**MAD** (Median Absolute Deviation) is a robust spread estimator:

```
   MAD = median( |xᵢ − median(x)| )
```

For our [3, 5, 7, 9, 11], the deviations from median are [4, 2, 0, 2, 4],
and the median of those is 2. So MAD = 2.

To convert MAD to a "standard-deviation-equivalent" σ for a Gaussian
distribution, multiply by `1.4826`:

```
   σ̂_robust = 1.4826 × MAD
```

The constant comes from the fact that for a true Gaussian,
`MAD = 0.6745 × σ`, and `1/0.6745 ≈ 1.4826`. With this scaling,
MAD-based σ converges to the true σ as N grows for clean Gaussian
data, and stays meaningful when up to half the data is garbage.

### How it's used here

- `src/litterbox/gas_anomaly.py` `_fit_log_gaussian()` (lines 90-111)
  computes `(median, MAD-based σ)` for a cat's gas readings.
- `src/litterbox/trend_anomaly.py` `_mad_sigma()` (lines 70-78) does
  the same for the trend detector's per-channel σ estimates.

Both detectors share the same robust statistics philosophy.

### Failure modes

- For very small samples (N < ~20), MAD has high variance and may
  underestimate σ. The detectors compensate with `min_visits_per_cat`
  gates that prevent firing until enough data is in.
- If the underlying distribution is heavy-tailed (lots of legitimately
  large readings), the 1.4826 conversion is wrong. Our data is
  approximately log-normal, so the conversion holds in log-space (next
  section).
- If MAD is exactly zero (all readings identical), σ̂ is zero and
  z-scores would be infinite. Both modules guard against this with a
  `_DEGENERATE_SIGMA` check that returns `None` instead.

---

## 4. Log-transforms — why we work in log-space for gas { #4-log-transforms }

### The problem

Gas sensor readings (NH₃ in ppb, CH₄ in ppb) span a huge range —
from near-zero up to several hundred. Their distribution is **right-skewed**:
most values are small, with a long tail of occasional high values.

If we fit a Gaussian directly to those raw values:

- The mean gets pulled up by the tail.
- The detector ends up flagging too many "normal" small readings (the
  Gaussian assumes symmetric variation around the mean, which isn't
  what raw ppb values do).
- The standard deviation σ has no clear meaning at the bottom of the
  range — "mean − 2σ" can easily go negative, which is impossible for
  a concentration.

### The fix: take the log

For data that's positive and right-skewed, `log(x)` often produces a
distribution that's much closer to symmetric (Gaussian-like). This
is called **log-normality** — a variable X is log-normal if log(X) is
normal.

We use `log1p(x) = log(1 + x)` instead of `log(x)` to handle the
edge case where x = 0 cleanly (`log(0)` is undefined; `log1p(0) = 0`).
For non-zero values, `log1p(x) ≈ log(x)` for `x >> 1`, so the
asymptotic behaviour is the same.

### The math

If `Y = log1p(X)` is approximately Gaussian with mean μ and σ in
log-space, then:

- The "typical" value of X (median) is `expm1(μ) = e^μ − 1`.
- A "k σ" deviation in log-space corresponds to a multiplicative
  factor of `e^(kσ)` in raw-ppb space — i.e. a 2σ event multiplies
  the typical value by `e^(2σ)`.
- Negative readings become impossible because `expm1(anything) ≥ −1`,
  and we clamp to 0.

This means a "2σ event" in log-Gaussian space corresponds to a
*relative* increase, not an absolute one. That's exactly what we want
for cats with very different baseline ammonia levels — Luna's "2σ
above normal" might be 80 ppb while Whiskers' is 30 ppb, and they
both correspond to the same statistical surprise.

### How it's used here

- `src/litterbox/gas_anomaly.py` `_fit_log_gaussian()` applies
  `math.log1p()` before computing median + MAD-σ (line 104).
- The z-score computation at line 119 uses `math.log1p(reading)`
  consistently, so everything is in log-space.
- `src/litterbox/history_plot.py` plots gas channels on a **log y-axis**
  (the design note in CLAUDE.md mentions this) so visual inspection
  matches the detector's space.
- `src/litterbox/trend_anomaly.py` does NOT log-transform. It works on
  raw ppb values for trend purposes because we're comparing two windows
  with similar distributions, not asking about absolute typicality.
  This is a deliberate choice: trend asks "did the mean shift?",
  per-visit asks "is this value typical?".

### Failure modes

- If the cat's gas distribution is truly bimodal or otherwise non-
  Gaussian even in log-space, MAD-σ in log-space is a poor model.
  In practice the readings are close enough to log-normal for this
  to work.
- For a cat with very low gas readings (near zero), the log1p
  transform compresses small values, so the detector becomes less
  sensitive there. This is fine — small differences at the bottom of
  the range aren't medically meaningful.

---

## 5. The gas anomaly detector — log-Gaussian z-scores { #5-gas-anomaly }

Now we can put pieces 3 and 4 together. The full per-visit gas
detector is:

### The model

Each cat's prior NH₃ readings are assumed log-normal. We fit:

```
   μ̂  = median(log1p(x₁), log1p(x₂), ..., log1p(xₙ))
   σ̂  = 1.4826 × MAD(log1p(x₁), ..., log1p(xₙ))
```

Same for CH₄ separately.

### Scoring a new visit

Given a new reading `x_new`, compute its z-score:

```
   z = (log1p(x_new) − μ̂) / σ̂
```

Translate to a tier:

| Condition | Tier | Approximate Gaussian tail probability |
|---|---|---|
| z < 2 | normal | 97.7% (one-sided) of normal data |
| 2 ≤ z < 3 | mild | 2.3% expected by chance |
| 3 ≤ z < 5 | significant | 0.13% expected |
| z ≥ 5 | severe | < 0.0001% expected |

The tier is computed for both NH₃ and CH₄, and the worse one becomes
the visit's `overall_tier`. We alarm only on the high side (positive z)
because high gas is the clinically interesting direction. Low gas is
either a clean cat or a sensor problem; either way it's not a health
concern.

### Per-cat vs pooled fallback

A new cat doesn't have its own history yet. The detector handles this
with a fallback chain:

1. **Per-cat fit**: if the cat has ≥10 prior non-null readings (config:
   `min_visits_per_cat`), use its own data.
2. **Pooled fit**: otherwise, if the population has ≥30 prior readings
   (config: `min_visits_pooled`), pool everything and use that.
3. **Insufficient data**: otherwise return `tier="insufficient_data"`
   and don't alarm.

This is a graceful degradation. New cats start under the population's
tent, then get their own personalized model once they have history.

### The "exclude self" trick

When scoring visit #142, we want to score it against ALL OTHER visits,
not against a pool that includes itself. This is critical for two
reasons:

1. Self-inclusion would bias the fit toward the new value, slightly
   lowering its own z-score and potentially hiding the anomaly.
2. After `record_exit` writes the new visit row, calling the scorer
   without exclusion would incorporate the not-yet-scored row into
   its own fit. The `exclude_visit_id` parameter in `_fetch_history`
   removes this circularity.

### How it's used here

- `src/litterbox/gas_anomaly.py` `score_gas_visit()` (line 164) is
  the public entry point. It returns a dict with z-scores, tiers,
  and which model fired.
- `src/litterbox/tools.py` `record_exit()` calls it after every
  visit's exit data is recorded.
- The verdict gets persisted on the visit row and surfaced in
  `get_anomalous_visits` and `get_visit_details`.

### Failure modes

- A cat with all-zero readings has `σ̂ = 0` → detector returns None
  for that channel. Reasonable behaviour: nothing to detect against.
- A cat with very few readings (5-9) falls back to pooled, which may
  be a poor match for its actual distribution. The tier is reported
  with `model_used="pooled"` so downstream consumers can adjust
  confidence.
- Slow drift over months gets absorbed (the median moves with the
  data). The trend detector is what catches that — see Section 9.

---

## 6. PCA / eigendecomposition — finding the shape of waveforms { #6-pca-eigen }

### The problem

Each visit produces a continuous time-series of weight readings — a
"waveform" showing the cat enter, settle, do its business, leave. We
resample every waveform to a fixed length L = 64 samples (so visits of
different durations are commensurable) and then ask: does this new
waveform look like the cat's typical visit pattern?

### The PCA idea (intuition first)

Imagine you've recorded 100 of Luna's visits as 64-dimensional vectors
(one number per resampled time point). Each visit lives as a single
point in 64-D space. Plot them all and you'd see they don't fill the
space evenly — they cluster along a few dominant directions. Maybe
the strongest direction captures "how big the weight bump is", the
second captures "how long the cat sits", the third "how active they
are during the visit", and so on.

PCA (Principal Component Analysis) finds those dominant directions
mathematically. The first principal component is the direction along
which the visits vary the most. The second is the direction along
which they vary the most subject to being orthogonal (perpendicular)
to the first. And so on.

Once you have those directions, you can project a new visit onto them
and ask: how much of this visit's structure is captured by the cat's
"typical" directions vs how much lives in directions the cat usually
doesn't go?

### The math

Let `X` be the K × L matrix of K visits, each a row of length L (after
zero-mean subtraction — see "DC term" below). The **sample covariance
matrix** is:

```
   C = (1 / (K − 1)) × Xᵀ X        (shape L × L)
```

This C is symmetric and positive semi-definite. Its **eigendecomposition**
finds vectors `v₁, v₂, ..., vₗ` and scalars `λ₁ ≥ λ₂ ≥ ... ≥ λₗ ≥ 0`
such that:

```
   C × vᵢ = λᵢ × vᵢ
```

Each `vᵢ` is a "principal direction" in L-dimensional space, and
`λᵢ` is the variance along that direction.

In code:
```python
eigenvalues, eigenvectors = np.linalg.eigh(C)
```

### Explained variance and component selection

The **fraction of total variance** explained by the first n components
is `(λ₁ + λ₂ + ... + λₙ) / (λ₁ + ... + λₗ)`. By choosing n large enough
that this fraction crosses a target (e.g. 95%), you get the smallest
subspace that captures most of the data's structure.

In our system, the target is `explained_variance_threshold = 0.95`.
But because the cluster detector (Section 8) needs **fixed-length**
coefficient vectors across all cats, we use `uniform_n = 4` after
empirically calibrating that 4 components achieve 95% coverage on
~95% of synthetic waveforms (see `eigen_sim_summary.md`).

### Scoring a new waveform

Given a fitted basis `V_N` (the top N eigenvectors as columns) and a
new waveform `x`:

1. Project: `coefficients = V_Nᵀ × x`  (length-N vector)
2. Reconstruct: `x_hat = V_N × coefficients`
3. Residual: `r = x − x_hat`
4. **Explained Variance (EV)** of the projection:

```
   EV = 1 − ‖r‖² / ‖x‖²
```

EV is in [0, 1]. EV = 1 means x is perfectly captured by the cat's
typical subspace; EV = 0 means it's orthogonal to that subspace
(completely unusual shape).

Tiers based on EV (from `td_config.json` `eigen.anomaly_thresholds`):

| EV | Tier |
|---|---|
| ≥ 0.90 | normal |
| ≥ 0.70 | mild |
| ≥ 0.40 | significant |
| < 0.40 | major |

### The DC term — why we subtract the mean

A waveform `x` has a "DC term" (its overall mean) and an "AC term"
(deviations from the mean). Without subtracting the DC term first, the
first principal component would always end up being "the direction
of the mean" — which is uninteresting and would dominate the
eigenvalues, leaving little room for the actually-interesting shape
information.

`eigen_analyser.py:160` computes `dc_term = float(np.nanmean(waveform))`
and stores it separately. The eigenanalysis is done on the zero-mean
residual. The DC term itself is recorded per visit and **shown in
reports** but no detector currently alarms on it (the trend detector
is what tracks DC drift).

### How it's used here

- `src/litterbox/eigen_analyser.py` is the implementation.
- `EigenAnalyser._analyse()` (lines ~130-260) is the main entry: NaN
  handling, DC subtraction, model selection (per-cat vs pooled),
  covariance, eigh, regularization (next section), projection, EV
  scoring.
- The verdict is stored in the `eigen_waveforms` table.
- Reports rendered by `eigen_query.py` show per-visit EV alongside
  the DC term and the coefficient vector.

### Failure modes

- **Overfitting**: with K samples and L dimensions, when K < L the
  covariance is rank-deficient and the eigendecomposition has L−K+1
  zero eigenvalues. The "natural" basis fits those K samples
  exactly. Tikhonov regularization (next section) is what prevents
  this from being garbage.
- **EV is a weak signal alone.** As the OC study showed, EV catches
  GROSS shape changes (80% energy swap → 97% TPR) but misses
  subtle ones (10% swap → 7% TPR). The cluster layer (Section 8)
  was added to catch those.
- **DC term not in any alarm.** A slowly drifting DC weight does not
  trip the eigen detector — the trend detector handles it.

---

## 7. Tikhonov regularization — taming small sample sizes { #7-tikhonov }

### The problem

When fitting a per-cat eigen model, K (number of waveforms) is often
smaller than L (length of waveform = 64). The covariance matrix is
then rank-deficient — there are directions in which no variation has
been observed, so the model says "variance there is exactly zero".
Any new waveform with structure in those directions gets infinite
z-score, which is meaningless.

### The fix

**Tikhonov regularization** (also known as ridge regression in another
context) adds a small positive constant to the diagonal of the
covariance matrix:

```
   C_reg = C + α × I
```

where I is the L×L identity and α is small. This makes the matrix
positive definite (all eigenvalues > 0) and has a smooth interpretation:
"assume there's at least α units of variance in every direction, on
top of what we observed".

In code (`eigen_analyser.py:201-204`):

```python
if model_type == "per_cat" and K < self._L:
    alpha = self._reg_eps * np.trace(C) / self._L
    C += alpha * np.eye(self._L)
    regularized = True
```

The α is set to `regularization_epsilon` (default 0.01) times the
mean diagonal of C. So it's small relative to the actual variances we
observed, but big enough to prevent zero eigenvalues.

### Why not always regularize

When K >> L (pooled model with hundreds of waveforms), the covariance
is full-rank and well-conditioned — regularization would just add
noise. We only apply it when K < L, where it's necessary.

### How it's used here

The regularized models are flagged with `regularized=True` in the
`eigen_models` table so reports can indicate that a verdict came from
a regularized fit (slightly less trustworthy than a full-rank one).

### The deeper picture

Regularization is one of the key ideas in machine learning. It comes up
in many forms: L2 penalty in linear regression, dropout in neural
networks, smoothing in density estimation. The unifying theme: when
your data is finite, the unconstrained "best fit" overfits noise.
Regularization injects a prior assumption ("smooth solutions are
better") that shrinks the fit toward something sensible.

---

## 8. Gaussian Mixture Models + BIC — finding clusters { #8-gmm-bic }

### The problem

The eigen detector tells you "this waveform's coefficient vector is
inside the cat's typical subspace" or "outside". But within the
subspace, a cat may have **multiple typical visit patterns** — maybe
short morning visits and long evening visits, or fast-pee visits vs
defecation visits. A new visit might project nicely into the subspace
(high EV) but land in a region the cat never visits (off all clusters).
That's still anomalous, just in a different way.

### The Gaussian Mixture Model idea

A Gaussian Mixture Model (GMM) says: this dataset is generated by k
different Gaussian distributions, each with its own mean and covariance,
mixed in proportions `π₁ + π₂ + ... + πₖ = 1`. Given a dataset, fit
the k Gaussians (their means, covariances, and weights) using the EM
algorithm.

For Luna's coefficient vectors, the fitted GMM might find:

- Cluster 1 (60% weight): morning visits, mean coefficients ≈ [0.5, -0.3, ...]
- Cluster 2 (40% weight): evening visits, mean coefficients ≈ [-0.2, 0.7, ...]

For a new visit, compute the **likelihood** under the fitted mixture:
high likelihood = "this point is consistent with one of Luna's
clusters", low likelihood = "this is in a region she never visits".

### The math

A GMM's probability density is:

```
   p(x) = Σᵢ πᵢ × N(x; μᵢ, Σᵢ)
```

where `N(x; μ, Σ)` is the multivariate normal with mean μ and
covariance Σ. The **log-likelihood** of a single point x is just
`log(p(x))` — a single negative number. Higher (less negative) =
"more typical".

We score a new point by its log-likelihood, then z-score it against
the distribution of log-likelihoods on the training data:

```
   z = (logL_new − mean(logL_train)) / std(logL_train)
```

A very negative z means "much less likely than the typical training
point" → anomalous.

Tiers (from `td_config.json` `cluster.z_score_thresholds`):

| z | Tier |
|---|---|
| ≥ −2 | normal |
| −3 ≤ z < −2 | mild |
| −4 ≤ z < −3 | significant |
| z < −4 | major |

(Note these are negative — we're alarming on points that are
*unlikely* under the model, so low likelihood = high alarm.)

### Choosing k — the BIC trick

How many clusters does this cat have? Too few (k=1) under-fits and
loses the cluster structure. Too many (k=10) over-fits and starts
labelling every visit its own cluster.

**BIC** (Bayesian Information Criterion) is a model-selection score
that trades fit quality against complexity:

```
   BIC = −2 × log-likelihood + p × log(N)
```

where p is the number of parameters in the model and N is the sample
size. Lower BIC is better. The penalty `p × log(N)` discourages
adding more clusters than the data justifies.

We sweep k from 1 to `max_clusters = 5`, fit a GMM for each, and pick
the one with the lowest BIC. If k=1 wins, the data has no detectable
cluster structure — the cat's visits are unimodal in coefficient
space, and we treat that as fine (no false alarms from the cluster
layer).

### How it's used here

- `src/litterbox/cluster_analyser.py` does this with scikit-learn's
  `GaussianMixture` class.
- `n_init = 5` means EM is restarted 5 times from different random
  starts and the best log-likelihood is kept (avoids local optima).
- The cluster model is stored in `cluster_models`; per-visit z-scores
  in `eigen_waveforms.cluster_z_score`.

### What the OC study found

From `simulator/oc_report.md`:

- BIC needs ≈200 samples to reliably pick the true k=2 in the synthetic
  test. Below 100 samples it usually collapses to k=1.
- Once converged, the cluster detector is sharp: 100% TPR at offset
  3σ from the cluster centroid, FPR 6% at ≥mild on held-out clean
  points.
- This is the workhorse anomaly detector in the eigen pipeline. EV
  alone catches very little; the cluster layer catches almost
  everything an EV-clean swap can sneak in.

### Failure modes

- BIC stability is data-dependent. With small N, k* is noisy across
  re-fits.
- GMM EM can converge to local optima. `n_init=5` mitigates but
  doesn't eliminate.
- Spherical / diagonal covariances (cheaper variants) would fail on
  rotated cluster shapes. We use full covariances.
- Like all parametric models, GMMs assume Gaussian-shaped clusters.
  Heavy-tailed or non-elliptical cluster shapes get poorly modelled.

---

## 9. The trend detector — Welch-style mean-shift z-test { #9-welch-trend }

### The problem

The per-visit detectors are blind to slow drifts. A cat losing 100 g
over six months has visits that each look fine in isolation. Only
the aggregate has shifted. We need a detector that explicitly compares
"now" to "before".

### The two-window setup

Split each cat's history into two windows:

- **Recent**: last 14 days (config: `days_recent`)
- **Baseline**: the 75 days immediately before that (config: `days_baseline`)

For each metric (cat weight, waste weight, NH₃ peak, CH₄ peak), compute
sample statistics for both windows.

### The z-test

The classical question "is the recent mean different from the baseline
mean?" is a textbook two-sample hypothesis test. We use a Welch-style
variant with **MAD-based σ** for robustness:

```
            (m_R − m_B)
   z = ──────────────────────
        √(σ²_B/n_R + σ²_B/n_B)
```

Where:
- `m_R, m_B` = mean of recent / baseline windows
- `n_R, n_B` = sample counts
- `σ_B` = MAD-based σ from the baseline window

We use `σ_B` for both terms (rather than `σ_R` for the recent term)
because the recent window is small (n_R ≈ 10) and its own σ estimate
is noisy. The baseline window has n_B ≈ 30+ which gives a more stable
σ estimate. Theoretically incorrect by Welch's exact formula but
practically more stable.

### Tiers

Same structure as the gas detector:

| |z| | Tier |
|---|---|
| < 2 | normal |
| 2 ≤ |z| < 3 | mild |
| 3 ≤ |z| < 5 | significant |
| ≥ 5 | severe |

### Direction-aware overlays

Different channels have different clinical meaning by direction:

- **Body weight**: alarm in either direction (gain OR loss) AND
  apply a clinical % threshold overlay (5/10/15% of baseline body
  weight). Take the worse of z-tier and pct-tier. The percentage
  overlay is what makes the detector clinically meaningful — a 5%
  weight loss is significant regardless of how statistically
  improbable it is.
- **NH₃ / CH₄**: alarm only on the high side (z > 0). Falling gas
  readings are good news.
- **Waste**: alarm only on the low side AND only under a constipation
  rule (described next).

### The constipation rule

Single low-waste visits are meaningless — cats often pee-only. A
constipation pattern is recognized only when ALL THREE conditions
fire:

1. **Frequency**: at least 50% of recent visits are "no-waste"
   (waste < 5g)
2. **Ratio**: the recent no-waste rate is at least 2× the baseline
   no-waste rate
3. **Volume**: the recent waste mean is significantly below baseline
   (z ≤ −2)

When all three fire, the waste channel tier is escalated to at least
`significant` (the constipation overlay).

### How it's used here

- `src/litterbox/trend_anomaly.py` `score_trends()` (line 244) is
  the public entry point.
- `tools.py` `get_trend_summary` formats it for one cat.
- `tools.py` `_scan_trend_alarms()` iterates all cats and surfaces
  the alarming ones in `get_anomalous_visits` and `get_trending_cats`.

### What the OC study found

- Per-channel ≥mild FPR on stable cats: 5–14% (matches calibrated
  2σ Gaussian tail per direction).
- Overall ≥significant FPR (any channel firing): 7.5%.
- Sensitivity to a 3% body-weight loss: 98% at ≥significant.
- Constipation rule TPR: 39% at 50% no-waste rate, 84% at 70%, 100%
  at 90%.

### Failure modes

- Mean-shift z-tests are sensitive to outliers in the recent window
  (since n_R is small). One unusual visit can pull the recent mean
  enough to trigger a false alarm.
- The baseline window assumes stationarity over 75 days — for a cat
  whose normal genuinely changes (diet change, surgery recovery), the
  baseline doesn't reflect current reality and the detector
  over-alarms.
- The constipation rule is conservative by design (3-condition AND).
  It misses early constipation that hasn't yet developed a clear
  no-waste pattern.

---

## 10. Operating characteristics — TPR, FPR, ROC, calibration { #10-oc }

### The two error types

Every detector makes two kinds of mistakes:

- **False positive** (FP, also "type I error"): the detector says
  "anomaly" when nothing was actually wrong. Wastes the user's
  attention.
- **False negative** (FN, also "type II error"): the detector says
  "normal" when something actually was wrong. Misses the problem.

You can't simultaneously minimize both. Lowering the alarm threshold
catches more real anomalies (better TPR) but also fires on more
normal cases (worse FPR). Raising it does the opposite.

The fundamental terms:

- **TPR** (True Positive Rate, also "sensitivity", "recall"):
  TP / (TP + FN). Of all real anomalies, what fraction did we
  catch?
- **FPR** (False Positive Rate, also "fall-out"):
  FP / (FP + TN). Of all normal cases, what fraction did we
  incorrectly flag?

The trade-off is captured in an **ROC curve** — TPR plotted against
FPR as the threshold varies. The closer to the top-left corner (high
TPR, low FPR), the better the detector.

### Why thresholds need calibration

The thresholds in `td_config.json` (e.g. `mild = 2.0` z-score) are
**design choices** that determine the operating point on the ROC
curve. They aren't right or wrong in isolation — they're right or
wrong relative to your tolerance for FP vs FN.

Our calibration philosophy:

- `severe` → almost zero FPR. Any severe alarm should be treated as
  real.
- `significant` → small FPR (~1% per channel). Worth investigating
  but expect some false alarms.
- `mild` → moderate FPR (~5% per channel). "Watch this", not "act
  on this".

The tier names are user-facing and the thresholds were picked to give
the calibrated tail probabilities listed above on Gaussian data.

### Why we synthesize "ground truth" data

To measure FPR, you need a dataset where you KNOW nothing is wrong.
Real cat data doesn't come with that label — the real data IS the
thing we're trying to characterize. So the OC study generates
synthetic data from a known generative model (Gaussian for gas,
parametric waveforms for time-domain), runs the live detector against
it, and measures rates.

This is honest about a key limitation: we're characterizing the
detector against the model, not against reality. If real cats follow
distributions different from our synthetic model, the FPR/TPR numbers
won't match. The synthetic model was chosen to be approximately
realistic (log-normal gas, similar means and variances to observed
data) but it's not a perfect substitute.

### How it's used here

- `simulator/oc_study.py` is the entire OC characterization. It
  runs convergence and operating-point experiments for all four
  algorithmic detectors (gas, eigen, cluster, trend) and emits
  `simulator/oc_report.md`.
- The report has tables of FPR per tier and TPR per (tier, anomaly
  magnitude) for each detector.
- Findings drove design decisions: e.g. the trend detector's `mild`
  tier was downgraded from "act on it" to "watch this" in the system
  prompt because the per-channel FPR is ~10%.

### Why this matters for software developers

Most software bugs are deterministic — given the same input, you get
the same wrong output. ML-style bugs are often **statistical** — the
detector "works" on average but has a 5% chance of being wrong on
any given case. Whether 5% is acceptable depends on the deployment.

Always ask: "What's the FPR? What's the TPR? What's the cost of each
error type for the user?" The answers determine whether the detector
is fit for purpose.

---

## 11. Why each algorithm was picked for its job { #11-why-each }

A summary of the design choices and what alternatives were rejected:

| Choice | Why | Alternative considered |
|---|---|---|
| CLIP for cat ID | Pretrained, zero-shot, runs locally for free, semantically meaningful embedding | Train a custom CNN per household (too much per-user friction); use a face-recognition library (cats aren't faces, accuracy poor) |
| Robust median + MAD | 50% breakdown point, doesn't get poisoned by historical anomalies in a chronically-monitored cat | Mean + std (gets poisoned); winsorized mean (less principled) |
| log1p transform on gas | Gas readings are right-skewed and positive; log makes them approximately Gaussian and gives multiplicative semantics | Raw values (would need a different distribution model); Box-Cox (fitting an additional parameter is overkill) |
| z-score tiers | Universal language across all detectors, easy to interpret, calibratable | Percentile thresholds (less interpretable); LR thresholds (less standard) |
| PCA for waveform analysis | Linear, interpretable, fast, well-understood, doesn't need labels | Autoencoders (need lots of data, less interpretable); DTW (no fixed coefficient vector for downstream clustering) |
| GMM + BIC for clustering | Soft cluster assignments, principled k-selection, gives a likelihood for free | k-means (no likelihood, hard assignments); DBSCAN (parameters hard to tune across cats) |
| Welch-style mean-shift for trends | Standard, well-understood; using MAD-σ keeps the family robust | EWMA (more complex, no clear advantage); CUSUM (designed for industrial monitoring, overkill) |
| Synthetic OC study for calibration | Honest about what we don't know, reproducible, no real cats harmed | Just trust theoretical Gaussian tail probabilities (would miss real-world deviations); A/B test on real cats (ethically questionable for a screening tool) |

The unifying themes are **robust statistics**, **interpretable models**, and
**explicit calibration**. None of the components are state-of-the-art for
their narrow problem (a custom CNN would beat CLIP on accuracy, an LSTM
might beat PCA on waveforms), but the combination is debuggable,
maintainable, and trustworthy.

---

## 12. Further reading { #12-further-reading }

If you want to go deeper on any of the algorithms here:

**Robust statistics**
- *Robust Statistics* by Huber & Ronchetti (2009) — the canonical
  reference for breakdown points and MAD-based estimators.

**CLIP and image embeddings**
- Radford et al., "Learning Transferable Visual Models From Natural
  Language Supervision" (2021) — the original CLIP paper.
- Reimers & Gurevych, "Sentence-BERT" (2019) — explains the embedding
  + cosine similarity pattern we use.

**PCA and eigendecomposition**
- *Pattern Recognition and Machine Learning* by Bishop (2006), Chapter 12
  — classic PCA derivation and probabilistic interpretation.
- *Numerical Linear Algebra* by Trefethen & Bau (1997) — for the
  numerical-stability side of eigendecomposition and SVD.

**Gaussian Mixture Models**
- Bishop (2006), Chapter 9 — GMM and the EM algorithm.
- Schwarz, "Estimating the Dimension of a Model" (1978) — the
  original BIC paper.

**Hypothesis testing**
- *All of Statistics* by Wasserman (2004), Chapter 10 — concise
  treatment of two-sample tests including Welch's.

**Anomaly detection generally**
- Chandola, Banerjee, Kumar, "Anomaly Detection: A Survey" (ACM
  Computing Surveys, 2009) — comprehensive overview of the field.

**Operating characteristics**
- *Pattern Classification* by Duda, Hart, Stork (2000), Chapter 1 — the
  clearest treatment of error trade-offs and ROC curves.

For the codebase itself, the test files (`tests/test_*_anomaly.py`)
are working examples of every algorithm here. Reading them with this
tutorial open is probably the fastest path to understanding the actual
implementation.
