make_sdrf_test_data <- function() {
  set.seed(33)
  psu <- rep(seq_len(40), each = 3)
  predictor <- matrix(rnorm(120 * 4), ncol = 4)
  response <- cbind(
    predictor[, 1] + rnorm(120),
    predictor[, 2] * rnorm(120)
  )
  list(
    X = predictor,
    Y = response,
    pi = rep(c(0.2, 0.4, 0.8), length.out = 120),
    psu = psu
  )
}

test_that("SDRF returns normalized distributions for a bivariate response", {
  data <- make_sdrf_test_data()
  fit <- sdrf(
    X = data$X,
    Y = data$Y,
    pi = data$pi,
    psu.id = data$psu,
    num.trees = 24,
    q = 0.5,
    min.obs = 5,
    lambda.max = Inf,
    num.rff = 64,
    bandwidth = 1,
    response.transform = "scale",
    rff.backend = "cpu",
    multiplier.method = "none",
    num.threads = 2,
    seed = 52,
    rff.seed = 53
  )

  prediction <- predict(fit, newdata = data$X[1:5, , drop = FALSE])
  expect_s3_class(fit, "sdrf")
  expect_identical(dim(prediction$weights), c(5L, 120L))
  expect_identical(dim(prediction$y), c(120L, 2L))
  expect_equal(as.numeric(rowSums(prediction$weights)), rep(1, 5),
               tolerance = 1e-12)
  expect_true(all(prediction$weights >= 0))
})

test_that("serialized leaf masses equal Algorithm 1 estimation masses", {
  data <- make_sdrf_test_data()
  fit <- sdrf(
    X = matrix(0, nrow = 120, ncol = 1),
    Y = data$Y,
    pi = data$pi,
    psu.id = data$psu,
    num.trees = 8,
    q = 0.5,
    min.obs = 5,
    lambda.max = Inf,
    num.rff = 16,
    bandwidth = 1,
    response.transform = "none",
    rff.backend = "cpu",
    multiplier.method = "none",
    num.threads = 1,
    seed = 71,
    rff.seed = 72
  )

  for (tree in seq_len(fit[["_num_trees"]])) {
    samples.by.node <- fit[["_leaf_samples"]][[tree]]
    masses.by.node <- fit[["_leaf_sample_weights"]][[tree]]
    for (node in seq_along(samples.by.node)) {
      if (length(samples.by.node[[node]]) == 0L) {
        next
      }
      # C++ sample IDs are zero-based.
      rows <- samples.by.node[[node]] + 1L
      expected <- 1 / ((1 - fit$sdrf.q) * data$pi[rows])
      expect_equal(masses.by.node[[node]], expected, tolerance = 1e-13)
    }
  }
})

test_that("survey and multiplier validation rejects ambiguous inputs", {
  data <- make_sdrf_test_data()
  expect_error(
    sdrf(data$X, data$Y, pi = rep(0, 120), psu.id = data$psu),
    "pi must"
  )
  expect_error(
    sdrf(
      data$X,
      data$Y,
      pi = data$pi,
      psu.id = data$psu,
      num.trees = 3,
      resampling.multipliers = matrix(1, 120, 2)
    ),
    "n by num.trees"
  )
  expect_error(
    sdrf(
      data$X,
      data$Y,
      pi = data$pi,
      psu.id = data$psu,
      num.trees = 3
    ),
    "Strict SDRF requires"
  )
  one.psu.per.tree <- matrix(0, nrow = 120, ncol = 2)
  one.psu.per.tree[data$psu == 1, 1] <- 1
  one.psu.per.tree[data$psu == 2, 2] <- 1
  expect_error(
    sdrf(
      data$X,
      data$Y,
      pi = data$pi,
      psu.id = data$psu,
      num.trees = 2,
      resampling.multipliers = one.psu.per.tree
    ),
    "at least two PSUs"
  )
})

test_that("SDRF requires explicit outer-sample predictors", {
  data <- make_sdrf_test_data()
  fit <- sdrf(
    data$X,
    data$Y,
    pi = data$pi,
    psu.id = data$psu,
    num.trees = 4,
    min.obs = 5,
    num.rff = 8,
    bandwidth = 1,
    response.transform = "none",
    rff.backend = "cpu",
    multiplier.method = "none",
    num.threads = 1,
    seed = 81
  )
  expect_error(predict(fit), "outer-sample newdata")
})

test_that("stratified PSU multipliers handle singleton numeric labels", {
  multipliers <- sdrf_multipliers(
    psu.id = c(5L, 5L, 9L, 9L),
    strata.id = c(1L, 1L, 2L, 2L),
    num.trees = 3,
    method = "stratified_psu",
    seed = 2
  )
  expect_equal(multipliers, matrix(1, nrow = 4, ncol = 3))
})

test_that("storage estimates expose the dominant allocations", {
  estimate <- sdrf_storage_estimate(
    n = 1000,
    num.trees = 100,
    num.rff = 64,
    q = 0.5,
    threads = 8,
    n.test = 20
  )
  expect_true(all(estimate$bytes > 0))
  expect_true("serialized.forest.approx" %in% estimate$component)
  expect_true("dense.prediction.upper" %in% estimate$component)
})