# Batch vs sliding-window localisation

Reproducible companion to **AER1513 Assignment 3** (Qilong Cheng, University of
Toronto) — batch maximum-a-posteriori estimation on SE(3) with a stereo camera
and an IMU, on the Starry Night dataset. Write-up and an interactive version:
<https://thejerrycheng.github.io/localization.html>

The assignment's estimator runs on a course-distributed dataset that is not
redistributed here. This repo carries the same estimator on the 2-D "Lost in the
Woods" problem — a wheeled robot with odometry, measuring range and bearing to
known landmarks — where the structure and the conclusions are identical and
everything runs from a clean checkout in about two minutes.

```bash
pip install numpy matplotlib
python scripts/tools/run_se3.py --seeds 3            # Assignment 3, on SE(3)
python scripts/tools/run_localization.py --seeds 4   # Assignment 2, planar
```

## Assignment 3: SE(3), stereo + IMU (3 seeds, 160 steps)

| window | translation RMS [m] | rotation RMS [mrad] | time for 160 steps [s] |
| --- | --- | --- | --- |
| 1 (a filter) | 0.152 | 24.5 | 0.3 |
| 2 | 0.183 | 28.8 | 0.5 |
| 5 | 0.209 | 34.0 | 1.1 |
| 10 | 0.186 | 33.3 | 2.6 |
| 20 | 0.140 | 30.6 | 4.5 |
| **batch** | **0.070** | **10.9** | **0.5** |

Batch is **2.2× more accurate in both translation and rotation** than the best
sliding window, and **9× cheaper** than a 20-step one. The window results are not
monotonic in the window size — at three seeds the spread between 2, 5 and 10 is
linearisation noise, not a trend — but all of them sit two to three times worse
than batch.

The gap is far wider than in the planar case (18 %) because of the observation
model: a stereo camera sees a landmark only within range and inside the field of
view, so at any instant there are typically three or four measurements against
six degrees of freedom. A filter must commit to a pose on that. Batch need not —
a landmark first seen at step 90 constrains the pose at step 40, because both are
unknowns in the same linear system.

## Results (4 seeds, 300 steps)

| window | position RMS [m] | heading RMS [rad] | time for 300 steps [s] | inside its own 3σ |
| --- | --- | --- | --- | --- |
| 1 (a filter) | 0.0385 | 0.0119 | 0.43 | 99.4 % |
| 2 | 0.0378 | 0.0118 | 0.62 | 99.5 % |
| 5 | 0.0362 | 0.0114 | 1.24 | 99.6 % |
| 10 | 0.0338 | 0.0111 | 2.31 | 99.6 % |
| 20 | 0.0335 | 0.0106 | 4.07 | 99.6 % |
| 50 | 0.0377 | 0.0109 | 9.44 | 99.6 % |
| **batch** | **0.0317** | **0.0098** | **0.53** | — |

**1. Batch is more accurate and cheaper.** Per *step* a smaller window is faster —
the information matrix is (6N)². But a sliding window does that solve at every
step, while batch does one solve ever. Over 300 steps a window of 50 costs 22× a
filter and 18× full batch, for a worse estimate. The sliding window earns its
place by running online, not by being cheap.

**2. A sliding window is not inherently over-confident.** The report finds the
window's 3σ bounds fail to cover the error while the batch bounds do, and
suspects a poor covariance estimate. Holding the true noise fixed and scaling
the covariance the estimator *believes*:

| assumed covariance | ×0.25 | ×0.5 | ×1 | ×2 | ×4 |
| --- | --- | --- | --- | --- | --- |
| errors inside 3σ | 74.1 % | 92.8 % | **99.6 %** | 100 % | 100 % |
| position RMS [m] | 0.0338 | 0.0338 | 0.0338 | 0.0337 | 0.0337 |

With the right Q and R the window is textbook-consistent. The estimate itself is
unchanged across the whole sweep — a covariance mismatch does not degrade the
estimate, it degrades the estimate's account of itself, which is worse, because
nothing downstream can detect it.

**3. Uncertainty tracks the landmark count**, reproducing the report's third
observation: the worst stretch of the trajectory is the one where nothing is in
view.

The JavaScript on the website (`assets/js/localization_engine.js` there) is the
same Gauss-Newton solve, including the marginal covariance of the newest pose.
