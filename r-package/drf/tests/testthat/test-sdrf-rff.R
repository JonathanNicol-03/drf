test_that("Gaussian RFFs are reproducible and support bivariate responses", {
  set.seed(10)
  response <- matrix(rnorm(80), ncol = 2)

  first <- sdrf_rff(
    response,
    num.rff = 128,
    kernel = "gaussian",
    bandwidth = 1.2,
    response.transform = "none",
    seed = 91,
    backend = "cpu",
    num.threads = 1
  )
  second <- sdrf_rff(
    response,
    num.rff = 128,
    kernel = "gaussian",
    bandwidth = 1.2,
    response.transform = "none",
    seed = 91,
    backend = "cpu",
    num.threads = 2
  )

  expect_s3_class(first, "sdrf_rff")
  expect_identical(dim(first$features), c(40L, 256L))
  expect_identical(dim(first$frequencies), c(128L, 2L))
  expect_equal(first$frequencies, second$frequencies, tolerance = 0)
  expect_equal(first$features, second$features, tolerance = 1e-14)
})

test_that("RFF inner products approximate both configured kernels", {
  response <- rbind(c(-1.0, 0.5), c(0.25, -0.75), c(1.5, 1.0))
  bandwidth <- 1.3

  gaussian <- sdrf_rff(
    response,
    num.rff = 12000,
    kernel = "gaussian",
    bandwidth = bandwidth,
    response.transform = "none",
    seed = 17,
    backend = "cpu",
    num.threads = 2
  )
  gaussian.approximation <- gaussian$features %*% t(gaussian$features)
  squared.distance <- as.matrix(stats::dist(response))^2
  gaussian.exact <- exp(-squared.distance / (2 * bandwidth^2))
  expect_equal(unname(gaussian.approximation), unname(gaussian.exact),
               tolerance = 0.035)

  laplacian <- sdrf_rff(
    response,
    num.rff = 12000,
    kernel = "laplacian",
    bandwidth = bandwidth,
    response.transform = "none",
    seed = 18,
    backend = "cpu",
    num.threads = 2
  )
  laplacian.approximation <- laplacian$features %*% t(laplacian$features)
  l1.distance <- as.matrix(stats::dist(response, method = "manhattan"))
  laplacian.exact <- exp(-l1.distance / bandwidth)
  expect_equal(unname(laplacian.approximation), unname(laplacian.exact),
               tolerance = 0.035)
})

test_that("windowed RFF preprocessing persists one object at a time", {
  response <- matrix(seq_len(60) / 10, ncol = 2)
  output.directory <- tempfile("sdrf-rff-")
  windows <- rbind(c(1, 15), c(6, 25))

  manifest <- sdrf_rff_windows(
    response,
    windows,
    output.dir = output.directory,
    num.rff = 16,
    kernel = "gaussian",
    bandwidth = 1,
    response.transform = "none",
    backend = "cpu",
    num.threads = 1,
    seed = 4
  )

  expect_equal(manifest$observations, c(15, 20))
  expect_true(all(file.exists(manifest$path)))
  expect_s3_class(readRDS(manifest$path[1]), "sdrf_rff")

  expect_error(
    sdrf_rff_windows(
      response[nrow(response):1, , drop = FALSE],
      windows,
      output.dir = output.directory,
      num.rff = 16,
      kernel = "gaussian",
      bandwidth = 1,
      response.transform = "none",
      backend = "cpu",
      num.threads = 1,
      seed = 4
    ),
    "does not match this response window"
  )
  expect_error(
    sdrf_rff_windows(
      response,
      windows,
      output.dir = output.directory,
      num.rff = 32,
      kernel = "gaussian",
      bandwidth = 1,
      response.transform = "none",
      backend = "cpu",
      num.threads = 1,
      seed = 4
    ),
    "different projection options"
  )
})

test_that("cache validation includes captured transform values", {
  response <- matrix(seq_len(24), ncol = 2)
  output.directory <- tempfile("sdrf-transform-")
  make_transform <- function(multiplier) {
    function(row) row * multiplier
  }

  sdrf_rff_windows(
    response,
    windows = matrix(c(1, 12), nrow = 1),
    output.dir = output.directory,
    num.rff = 8,
    kernel = "gaussian",
    bandwidth = 1,
    response.transform = make_transform(1),
    backend = "cpu",
    num.threads = 1,
    seed = 5
  )
  expect_error(
    sdrf_rff_windows(
      response,
      windows = matrix(c(1, 12), nrow = 1),
      output.dir = output.directory,
      num.rff = 8,
      kernel = "gaussian",
      bandwidth = 1,
      response.transform = make_transform(2),
      backend = "cpu",
      num.threads = 1,
      seed = 5
    ),
    "different projection options"
  )
})