make_drf_options_test_data <- function() {
  set.seed(94)
  list(
    X = matrix(rnorm(80), ncol = 2),
    Y = matrix(rnorm(80), ncol = 2)
  )
}

fit_drf_options_test <- function(data, num.trees, ci.group.size) {
  drf(
    data$X,
    data$Y,
    num.trees = num.trees,
    ci.group.size = ci.group.size,
    splitting.rule = "CART",
    min.node.size = 2,
    honesty = FALSE,
    num.threads = 1,
    seed = 95
  )
}

test_that("ordinary DRF validates and rounds tree groups", {
  data <- make_drf_options_test_data()
  default.small.forest <- drf(
    data$X,
    data$Y,
    num.trees = 10,
    splitting.rule = "CART",
    min.node.size = 2,
    honesty = FALSE,
    num.threads = 1,
    seed = 95
  )

  expect_equal(default.small.forest[["ci.group.size"]], 1L)
  expect_equal(default.small.forest[["_num_trees"]], 10L)
  expect_error(
    fit_drf_options_test(data, num.trees = 0, ci.group.size = 1),
    "num.trees must be one finite positive whole number"
  )
  expect_error(
    fit_drf_options_test(data, num.trees = 10, ci.group.size = 0),
    "ci.group.size must be one finite positive whole number"
  )
  expect_error(
    fit_drf_options_test(data, num.trees = Inf, ci.group.size = 1),
    "num.trees must be one finite positive whole number"
  )
  expect_error(
    fit_drf_options_test(data, num.trees = c(10, 20), ci.group.size = 1),
    "num.trees must be one finite positive whole number"
  )
  expect_error(
    fit_drf_options_test(data, num.trees = 10, ci.group.size = 1.5),
    "ci.group.size must be one finite positive whole number"
  )

  exact.multiple <- fit_drf_options_test(
    data, num.trees = 6, ci.group.size = 3
  )
  rounded.multiple <- fit_drf_options_test(
    data, num.trees = 100, ci.group.size = 3
  )
  expect_equal(exact.multiple[["_num_trees"]], 6L)
  expect_equal(rounded.multiple[["_num_trees"]], 102L)
})