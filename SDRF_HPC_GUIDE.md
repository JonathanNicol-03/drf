# SDRF Algorithm 1: implementation and Hypatia guide

## 1. Scope and research status

This branch adds the root-level PSU-honest Survey-calibrated Distributional
Random Forest (SDRF) described by Algorithm 1 of Zou, Matabuena, and Kosorok,
arXiv:2512.08179v2. It retains the existing DRF interface and adds a separate
`sdrf()` interface so existing analyses do not silently change meaning.

Implemented:

- One global Random Fourier Feature (RFF) projection per estimation window.
- Gaussian RBF and product-Laplacian kernels.
- Numeric multivariate responses, including the original bivariate use case.
- User-supplied, design-respecting resampling multipliers `n_star` for every
  tree and sampled unit.
- Algorithm 1 PSU-level split/estimation honesty.
- Split weights `n_star / (q * pi)`.
- Estimation weights `n_star / ((1 - q) * pi)`.
- Design-weighted kernel means and the SDRF MMD split score.
- Raw `min.obs`, within-child `lambda.max`, and optional maximum depth.
- Hajek-normalized estimation distributions at prediction time.
- Rolling/expanding-window RFF persistence and cache validation.
- CPU tree parallelism and an optional R torch/CUDA RFF backend.

Not implemented:

- Algorithm 2 (depth-adaptive honesty).
- A universal pseudo-population bootstrap generator for arbitrary survey
  designs. Exact multipliers must be supplied by the researcher.
- CI-group uncertainty weights for SDRF.
- GPU tree construction. CUDA is used only for optional matrix projection.

The paper has a v3 dated 26 July 2026. This code deliberately follows v2, as
requested. Reconcile v2 and v3 before freezing the dissertation's final method.

## 2. Architecture

```mermaid
flowchart TD
  A[R: validate X, Y, pi, PSU, strata] --> B[R: transform Y inside window]
  B --> C[R/C++ or torch: one RFF projection]
  C --> D[/sharedscratch optional RDS cache]
  D --> E[Rcpp sdrf_train]
  E --> F[C++ shared immutable Data]
  F --> G[C++ parallel trees]
  G --> H[PSU Bernoulli q honesty]
  H --> I[weighted RFF-MMD splits]
  I --> J[estimation-only weighted leaves]
  J --> K[serialized SDRF forest]
  K --> L[outer-sample X]
  L --> M[conditional training-response weights]
  M --> N[distributional scores and portfolio decisions]
```

The projection is attached once to `Data` and shared read-only by all tree
threads. Each complex feature is represented by adjacent real and imaginary
columns. This avoids a dependency on `std::complex` layout when a CUDA result is
copied to host memory.

For `D` complex features, the stored map is

```text
[Re(phi_1), Im(phi_1), ..., Re(phi_D), Im(phi_D)]
```

with

```text
phi_j(y) = exp(i * omega_j' y) / sqrt(D).
```

The supported kernels are defined exactly as follows:

```text
Gaussian:  k(y,z) = exp(-||y-z||_2^2 / (2 * bandwidth^2))
            omega coordinates ~ Normal(0, 1 / bandwidth^2)

Laplacian: k(y,z) = exp(-||y-z||_1 / bandwidth)
            omega coordinates ~ Cauchy(0, 1 / bandwidth)
```

`laplacian` therefore means the product/L1 Laplacian kernel. An isotropic L2
Laplacian kernel has a different multivariate spectral law and is not silently
substituted.

## 3. Installation on Hypatia

The cluster documentation recommends installing R in Conda. Keep source,
environments, and small configuration files in `/home`; use `/sharedscratch`
for projections, multiplier matrices, forests, and prediction artifacts.

### 3.1 Create an environment

Package names and available R versions can change. The following is a template:

```bash
module load anaconda

conda create -p "$HOME/conda-envs/sdrf" -c conda-forge \
  r-base=4.4 r-rcpp r-rcppeigen r-matrix r-fastdummies r-transport \
  r-devtools r-roxygen2 r-testthat make gxx_linux-64

conda activate "$HOME/conda-envs/sdrf"
```

If Hypatia uses a different Anaconda module name, inspect `module avail`. Do not
install or compile on a busy login node when the site asks for a debug job.

### 3.2 Optional Zen 4 optimization

The standard CPU nodes use AMD EPYC 9634 (Zen 4). Put host-specific flags in a
user Makevars file rather than in the package, preserving portability:

```make
# $HOME/.R/Makevars-hypatia
CXX11FLAGS = -O3 -march=znver4 -mtune=znver4 -pipe
```

Then build with:

```bash
export R_MAKEVARS_USER="$HOME/.R/Makevars-hypatia"
```

`-march=znver4` enables instructions supported by that CPU, including applicable
AVX-512 features. Do not distribute that binary to older processors.

### 3.3 Build, document, install, and test

Git symlinks under `r-package/drf/src` connect the R package to `bindings` and
`core/src`. Clone or transfer the repository in a way that preserves symlinks.

```bash
cd "$HOME/src/drf/r-package"
Rscript build_package.R

# Explicit test command after installation:
Rscript -e 'testthat::test_package("drf")'
```

The build script runs roxygen and `Rcpp::compileAttributes()`, so generated
`NAMESPACE`, `man/*.Rd`, and Rcpp registration should match the source. For a
manual build:

```bash
cd "$HOME/src/drf/r-package"
Rscript -e 'roxygen2::roxygenise("drf"); Rcpp::compileAttributes("drf")'
R CMD INSTALL drf
```

### 3.4 Optional CUDA projection

The optional backend uses the public API of the R `torch` package. It needs a
CUDA-enabled libtorch installation compatible with the node driver. Install and
test it on a GPU allocation, following local module policy:

```r
install.packages("torch")
torch::install_torch()
stopifnot(torch::cuda_is_available())
```

The projected matrix is always copied back to host RAM because C++ tree training
is CPU-based. For a bivariate response, CPU projection will often beat GPU after
launch, PCIe, and disk costs. Benchmark before reserving an A30.

## 4. Survey design inputs

Survey variables are not predictor columns unless they are also scientifically
meaningful covariates. Pass them separately to `sdrf()` and preserve row order.

### `pi`

- One first-order inclusion probability per training row.
- Must be finite and in `(0, 1]`.
- Pass `pi`, not the design weight. If a supplied weight is exactly `1/pi`,
  invert it first.
- Calibration, nonresponse, and post-stratification weights are not necessarily
  inverse first-order inclusion probabilities. Do not relabel them as `pi`
  without a design argument.

### `psu.id`

- One Primary Sampling Unit label per row.
- All rows in a PSU are routed to the same split or estimation set in a tree.
- Labels may repeat across strata; the R layer combines PSU and stratum labels.
- If `NULL`, each row is treated as its own PSU. That is valid only for an
  unclustered design.

### `strata.id`

- One stratum label per row.
- Used by convenience multiplier generation and PSU identity.
- Exact external multipliers must already respect stratum-specific sampling.

### `resampling.multipliers`

- Numeric `n x B` matrix.
- Entry `(i,b)` is `n_star[b,i]`, not a final survey weight.
- Values must be finite and nonnegative.
- Every column must activate at least two PSUs so Algorithm 1 honesty can form
  both sets.
- Matrix rows must match `X`, `Y`, `pi`, PSU, and strata order exactly.

For each tree, Algorithm 1 flips every active PSU exactly once. The code does
not redraw indicators to force a usable partition, because that would change
the Bernoulli law. If all `m` active PSUs enter the split set, which has
probability `q^m`, the tree has no estimation distribution and training stops
with an explicit error. An all-estimation draw has probability `(1-q)^m` and is
retained as a valid stump. Select `q` and ensure enough active PSUs per
multiplier column that `q^m` is negligible across `B` trees; changing the tree
seed is appropriate after recording the failed run.

The strict default is `multiplier.method="external"`. This is intentional. The
first-order probabilities, PSU IDs, and strata do not uniquely identify PPS,
SRSWOR at a second stage, finite-population corrections, or nonresponse steps.
Generate multipliers by reapplying the documented original design, ideally with
validated survey-bootstrap software, and save the matrix or a path per window.

The convenience methods are useful for smoke tests or explicitly matching
simple designs:

```r
# No design resampling. Explicit opt-in, mainly for diagnostics.
m_none <- sdrf_multipliers(psu, strata, B, method = "none")

# Independent mean-one unit multipliers.
m_poisson <- sdrf_multipliers(
  psu, strata, B, method = "poisson_unit", averages = 10, seed = 1
)

# Whole-PSU with-replacement draws within strata.
m_psu <- sdrf_multipliers(
  psu, strata, B, method = "stratified_psu", averages = 10, seed = 1
)
```

These convenience methods are not claimed to reproduce an arbitrary complex
design or the full supplementary pseudo-population algorithm.

### Asset-pricing interpretation

Equity panels are not automatically complex surveys. Before using survey
calibration, state the population, sampling mechanism, and inclusion
probabilities. If the data are the observed security universe after deterministic
filters, valid `pi` may not exist.

Using firm ID as `psu.id` can still enforce firm-level honesty over a rolling
window, but that is a cluster-honesty modeling choice, not evidence of survey
design calibration. With `pi=1`, describe the method accordingly. Do not infer
unknown inclusion probabilities from market capitalization without a sampling
design that justifies them.

## 5. RFF preprocessing

### 5.1 One window

```r
library(drf)

projection <- sdrf_rff(
  Y_train,
  num.rff = 512,
  kernel = "gaussian",
  bandwidth = NULL,
  response.transform = "scale",
  seed = 202601,
  backend = "cpu",
  num.threads = 16,
  output.file = "/sharedscratch/USER/sdrf/rff/window-001.rds"
)
```

`response.transform="scale"` estimates center and scale only from that training
window. Never scale with the outer test period. A function can be supplied, but
it must return a fixed finite numeric vector for every row.

The default bandwidth is the median pairwise Euclidean distance for Gaussian
and median pairwise Manhattan distance for product Laplacian. At most 2,500
deterministically spaced rows are used. Treat bandwidth as a tuned parameter in
the final study rather than relying only on the heuristic.

### 5.2 Rolling or expanding windows

`windows` is either a list of training-row indices or a two-column matrix of
inclusive start/end rows:

```r
window_ranges <- rbind(
  c(1, 120000),
  c(5001, 125000),
  c(10001, 130000)
)

manifest <- sdrf_rff_windows(
  Y = panel$Y,
  windows = window_ranges,
  output.dir = "/sharedscratch/USER/sdrf/rff/gaussian-512",
  prefix = "expanding",
  num.rff = 512,
  kernel = "gaussian",
  bandwidth = 0.85,
  response.transform = "scale",
  seed = 202601,
  backend = "cpu",
  num.threads = 16
)
```

Existing files are reused only when the response MD5, row order, kernel,
bandwidth, feature count, seed, and transform signature match. Otherwise the
call stops and asks for `overwrite=TRUE`.

Reusing one projection is valid across an `mtry`, `min.obs`, `lambda.max`, `q`,
or tree-count grid. It is not valid across kernel, bandwidth, transform,
`num.rff`, or frequency-seed values.

For nested time-series tuning, each inner training fold needs its own fitted
response transformation. A projection based on the complete outer training
window leaks inner validation responses into scaling, even though it does not
use the outer test period.

## 6. Train and predict

### 6.1 Strict Algorithm 1 example

```r
library(drf)

# All objects below must have the same row order.
multiplier_matrix <- readRDS(
  "/sharedscratch/USER/sdrf/multipliers/window-001.rds"
)

fit <- sdrf(
  X = X_train,
  Y = Y_train,
  pi = pi_train,
  psu.id = psu_train,
  strata.id = strata_train,
  resampling.multipliers = multiplier_matrix,
  num.trees = ncol(multiplier_matrix),
  q = 0.5,
  mtry = NULL,
  min.obs = NULL,
  lambda.max = NULL,
  max.depth = 12,
  rff = "/sharedscratch/USER/sdrf/rff/window-001.rds",
  num.threads = 168,
  seed = 41001
)

distribution <- predict(fit, newdata = X_outer)
```

`distribution$weights` has one row per outer predictor and one column per
training response. `distribution$y` is the original, untransformed `Y_train`.
For test point `x`, the estimated conditional law is the weighted empirical
distribution

```text
sum_i distribution$weights[x,i] * delta_{Y_train[i]}.
```

Weights are nonnegative and sum to one, up to floating-point tolerance.

### 6.2 Built-in functionals

```r
conditional_mean <- predict(fit, X_outer, functional = "mean")
conditional_sd <- predict(fit, X_outer, functional = "sd")

conditional_quantiles <- predict(
  fit,
  X_outer,
  functional = "quantile",
  quantiles = c(0.01, 0.05, 0.5, 0.95, 0.99)
)

conditional_covariance <- predict(fit, X_outer, functional = "cov")

loss_probability <- predict(
  fit,
  X_outer,
  functional = "custom",
  custom.functional = function(y, w) sum(w[y[, 1] < 0])
)
```

For a bivariate response, preserve the joint row weights. Computing each margin
separately discards conditional dependence. A scenario can be drawn by sampling
a training row with probabilities from the relevant weight row.

SDRF does not expose ordinary OOB predictions. Always supply an outer-sample
`newdata`, and keep its realized `Y_outer` separate until scoring.

## 7. Outer-sample evaluation

### 7.1 Correct rolling protocol

For each forecast origin:

1. Define the outer training and test dates before looking at outcomes.
2. Restrict `X`, `Y`, and every design vector to the training indices.
3. Recreate or load design-respecting multiplier columns for that window.
4. Tune kernel, bandwidth, `num.rff`, `q`, `mtry`, `min.obs`, `lambda.max`,
   depth, and tree count using time-ordered inner validation only.
5. Refit the chosen configuration on the entire outer training window.
6. Predict the untouched outer `X`.
7. Score against outer `Y` and construct the portfolio using information
   available at that forecast origin.
8. Save compact forecasts, scores, decisions, and provenance.

Do not randomly shuffle finance observations across inner folds. Account for
overlapping returns and serial dependence in uncertainty estimates.

### 7.2 Proper distributional scores

For one-dimensional returns, prioritize Continuous Ranked Probability Score
(CRPS). For a weighted empirical forecast `(y_i, w_i)` and realization `y`:

```text
CRPS = sum_i w_i |y_i - y|
       - 0.5 * sum_i sum_j w_i w_j |y_i - y_j|.
```

For a multivariate response, use the energy score:

```text
ES = sum_i w_i ||y_i - y||_2
     - 0.5 * sum_i sum_j w_i w_j ||y_i - y_j||_2.
```

Also report:

- Pinball loss at economically relevant quantiles.
- Empirical coverage and average width of central prediction intervals.
- Probability Integral Transform diagnostics for continuous margins.
- Kernel score or held-out MMD for joint-distribution comparison.
- Mean squared/absolute error only as conditional-mean benchmarks.

Raw log score is not defined usefully for an unsmoothed discrete empirical
forecast at a new continuous realization. Use a documented density smoother if
log score is required, and tune it without using the outer outcomes.

### 7.3 Tail-risk diagnostics

For loss-tail applications, report:

- VaR exceedance rate and unconditional coverage.
- Independence/conditional coverage of exceedances.
- Expected Shortfall loss or a joint elicitable VaR/ES score.
- Tail probability calibration by bins.
- Performance during predeclared stress periods.

Use block/bootstrap or HAC inference for time-dependent score differences.
Avoid interpreting many asset-date rows as independent observations.

### 7.4 Economic evaluation

Convert each conditional distribution into the exact decision rule declared in
advance. Depending on the dissertation design, report:

- Out-of-sample certainty-equivalent return under the chosen utility.
- Mean return, volatility, Sharpe and Sortino ratios.
- Maximum drawdown and tail loss.
- Turnover, leverage, concentration, and transaction costs.
- Constraint violations and infeasible optimization frequency.
- Differences against mean-only RF, historical, and linear/factor benchmarks.

Use the joint distribution when optimizing a multivariate payoff. Include all
transaction-cost and rebalancing assumptions in the outer loop, not afterward.

If the outer evaluation set is itself a probability sample, distinguish the
conditional SDRF training weights from evaluation-design weights. Aggregate
per-observation scores with the outer design weights when estimating a
population-level forecast score.

## 8. Parallel and GPU workflow on Hypatia

### 8.1 Tree building decision

Use the existing C++ backend, not joblib:

- Trees are already distributed across `std::async` workers.
- `num.threads` controls on-node parallelism.
- MPI is unnecessary because each forest is confined to one 168-core node.
- OpenMP should not be layered over the tree workers.
- SLURM arrays distribute independent windows or grid points across nodes.

Set BLAS/OpenMP libraries to one thread so they do not multiply the C++ workers:

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

Do not assume 168 workers is fastest. Weighted split scans and shared RFF reads
can become memory-bandwidth limited. Benchmark 42, 84, and 168 workers on one
representative window and use elapsed time, peak RSS, and throughput to choose.

### 8.2 Optional A30 projection job

```bash
#!/bin/bash
#SBATCH --job-name=sdrf-rff
#SBATCH --partition=gpu.A30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=/sharedscratch/%u/sdrf/logs/rff-%j.out

set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$HOME/conda-envs/sdrf"

export OMP_NUM_THREADS=1
Rscript "$HOME/project/precompute_windows.R"
```

Use `backend="torch"` only after a CPU/GPU benchmark. The `auto` backend uses a
conservative work threshold, but workload, CUDA startup, and GPFS performance
still determine the winner.

### 8.3 Throttled CPU array

```bash
#!/bin/bash
#SBATCH --job-name=sdrf-train
#SBATCH --partition=large-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=168
#SBATCH --mem=350G
#SBATCH --time=01:30:00
#SBATCH --array=0-19%3
#SBATCH --output=/sharedscratch/%u/sdrf/logs/train-%A_%a.out

set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$HOME/conda-envs/sdrf"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

Rscript "$HOME/project/train_batch.R" \
  "$SLURM_ARRAY_TASK_ID" "$SLURM_CPUS_PER_TASK"
```

If one array element owns a batch of 20 windows, run those forests sequentially
inside the element while each forest uses `SLURM_CPUS_PER_TASK`. Do not launch
20 simultaneous 168-thread R processes on one node. Alternatively, request
fewer CPUs per process and deliberately pack a measured number of processes.

Validate one small job on the debug partition before the full array. The exact
debug limits and account/QOS flags are site policy and should be confirmed with
the Hypatia documentation.

## 9. Storage and local-analysis decision

Use:

- `/home`: source, Conda environment, scripts, small manifests.
- `/sharedscratch`: RFF matrices, multiplier matrices, forest checkpoints,
  predictions, logs, and intermediate tuning results.
- node-local scratch (`$TMPDIR` or `$SLURM_TMPDIR`, if provided): temporary
  files that do not need to survive the job.

Uncompressed RDS is intentional for large numerical arrays on fast flash: it
reduces CPU time and peak temporary memory. Archive final compact results with
compression after the array completes.

Estimate one window before submission:

```r
sdrf_storage_estimate(
  n = nrow(X_train),
  num.trees = 500,
  num.rff = 512,
  q = 0.5,
  threads = 168,
  n.test = nrow(X_outer)
)

# Measure the real object after a pilot fit.
format(object.size(fit), units = "GiB")
```

Useful first-order formulas are:

```text
RFF matrix                 about 16 * n * D bytes
multiplier matrix          about  8 * n * B bytes
R + C++ multiplier peak    about 16 * n * B bytes
concurrent tree weights    about 16 * n * min(threads, B) bytes
serialized forest          about 16 * n * (1-q) * B bytes plus tree structure
dense prediction weights   about  8 * n * n_test bytes
```

The returned prediction matrix is sparse, so its real size can be much lower
than the dense upper bound. Measure it with `object.size()`.

### Recommendation

Perform training, raw distribution prediction, CRPS/energy/tail scoring, and
portfolio construction on Hypatia. Forests and raw weight matrices can grow to
hundreds of MB or many GB per window, and local analysis would duplicate both
storage and RAM requirements.

Download compact artifacts instead:

- keys such as date and security ID;
- realized outer response;
- conditional means, variances, quantiles, tail probabilities;
- per-observation proper scores;
- portfolio holdings, returns, turnover, and constraints;
- chosen hyperparameters, seeds, code revision, and window boundaries.

Download a forest only when a pilot `object.size(fit)` is comfortably below the
transfer budget and local RAM is at least roughly three times that size. The
exact decision needs `n`, `B`, `D`, outer rows, and measured sparsity; the helper
above makes that decision reproducible.

## 10. Reproducibility and provenance

Record for every outer window:

- training and test row/date boundaries;
- Git commit or source archive checksum;
- package and R session versions;
- kernel, bandwidth, `num.rff`, response transform, and RFF seed;
- `q`, `mtry`, `min.obs`, `lambda.max`, depth, trees, and tree seed;
- multiplier-generation software, method, design parameters, and seed;
- MD5/path for RFF and multiplier inputs;
- SLURM job ID, CPUs, GPU/backend, elapsed time, and peak memory.

Code provenance:

- The repository is the existing GPL-3 DRF package. Its README states that it
  originated from DRF and earlier GRF/ranger code; those existing sources were
  modified in place.
- The SDRF-specific C++ and R code in this change was written for this project.
  No external repository snippets were copied or adapted.
- The implementation follows mathematical definitions in the DRF JMLR paper
  and SDRF arXiv v2 paper; equations are method provenance, not copied software.
- Optional GPU execution calls the documented public API of the R `torch`
  package. No torch source code is included or adapted.
- Test infrastructure uses `testthat` through its public API.

## 11. Final pre-run checklist

- [ ] Confirm whether SDRF paper v2 or v3 is the dissertation specification.
- [ ] Document the actual asset-sampling design and the meaning of `pi`.
- [ ] Validate multipliers against that design; do not rely on convenience
      methods without a matching design argument.
- [ ] Verify row-order MD5 and dimensions for every cached RFF/multiplier file.
- [ ] Run package tests on Linux with the same compiler used for production.
- [ ] Compare exact kernels with RFF approximations at selected `D` values.
- [ ] Benchmark CPU versus A30 projection on one representative window.
- [ ] Benchmark 42, 84, and 168 C++ workers and record peak RSS.
- [ ] Run one debug window end to end, including prediction and scoring.
- [ ] Measure forest and sparse prediction sizes before the full array.
- [ ] Keep outer outcomes unavailable to scaling, tuning, and portfolio fitting.