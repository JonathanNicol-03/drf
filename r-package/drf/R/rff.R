#' Precompute random Fourier features for an SDRF training window
#'
#' Builds the finite-dimensional complex feature map used by the SDRF MMD
#' split statistic. The output matrix stores each complex feature in adjacent
#' real and imaginary columns. Projection is applied only to the response
#' \code{Y}; predictors remain unchanged and are still used to choose split
#' variables and thresholds.
#'
#' @param Y Numeric response matrix. Any number of response columns is allowed,
#'   including the bivariate response used by the original DRF package.
#' @param num.rff Number of complex random Fourier features.
#' @param kernel Either \code{"gaussian"} for the Gaussian RBF kernel or
#'   \code{"laplacian"} for the product Laplacian kernel based on L1 distance.
#' @param bandwidth Positive kernel bandwidth. If NULL, a sampled median
#'   pairwise-distance heuristic appropriate to \code{kernel} is used.
#' @param response.transform Either \code{"scale"}, \code{"none"}, or a
#'   function mapping one response row to a numeric vector. Transformations are
#'   estimated and applied within this window to avoid look-ahead leakage.
#' @param seed Integer seed used to draw Fourier frequencies.
#' @param backend Projection backend: \code{"cpu"}, \code{"torch"}, or
#'   \code{"auto"}. The torch backend requires the optional R torch package
#'   and a CUDA-enabled installation.
#' @param num.threads Number of CPU threads. NULL uses the package default.
#' @param output.file Optional \code{.rds} path. On an HPC system, use a path
#'   under \code{/sharedscratch}; the object is saved without compression to
#'   reduce CPU time and temporary memory.
#'
#' @return An object of class \code{sdrf_rff} containing \code{features},
#'   \code{frequencies}, kernel metadata, transformation metadata, and a
#'   lightweight fingerprint of the untransformed response.
#' @export
sdrf_rff <- function(Y,
                     num.rff = 256L,
                     kernel = c("gaussian", "laplacian"),
                     bandwidth = NULL,
                     response.transform = c("scale", "none"),
                     seed = 1L,
                     backend = c("auto", "cpu", "torch"),
                     num.threads = NULL,
                     output.file = NULL) {
  if (is.vector(Y, mode = "numeric")) {
    Y <- matrix(Y, ncol = 1L)
  }
  validate_Y(Y, NROW(Y))
  Y <- as.matrix(Y)

  if (length(num.rff) != 1L || !is.numeric(num.rff) ||
      !is.finite(num.rff) || num.rff < 1 || num.rff != floor(num.rff)) {
    stop("num.rff must be one positive integer.")
  }
  num.rff <- as.integer(num.rff)
  kernel <- match.arg(kernel)
  backend <- match.arg(backend)
  num.threads <- validate_num_threads(num.threads)
  if (length(seed) != 1L || !is.numeric(seed) || !is.finite(seed) ||
      seed < 0 || seed > .Machine$integer.max || seed != floor(seed)) {
    stop("seed must be one integer between 0 and .Machine$integer.max.")
  }

  transformed <- transform_rff_response(Y, response.transform)
  Y.transformed <- transformed$values
  if (is.null(bandwidth)) {
    bandwidth <- estimate_rff_bandwidth(Y.transformed, kernel)
  }
  if (length(bandwidth) != 1L || !is.numeric(bandwidth) ||
      !is.finite(bandwidth) || bandwidth <= 0) {
    stop("bandwidth must be one finite positive number.")
  }

  # Gaussian RBF and product-Laplacian kernels differ only in their spectral
  # draw. Keeping frequencies in R makes CPU and CUDA paths exactly comparable.
  set.seed(as.integer(seed))
  response.dimension <- NCOL(Y.transformed)
  if (kernel == "gaussian") {
    frequencies <- matrix(
      stats::rnorm(num.rff * response.dimension, sd = 1 / bandwidth),
      nrow = num.rff,
      ncol = response.dimension
    )
  } else {
    frequencies <- matrix(
      stats::rcauchy(num.rff * response.dimension, scale = 1 / bandwidth),
      nrow = num.rff,
      ncol = response.dimension
    )
  }

  selected.backend <- select_rff_backend(
    backend,
    NROW(Y.transformed) * num.rff * response.dimension
  )
  if (selected.backend == "torch") {
    features <- project_rff_torch(Y.transformed, frequencies)
  } else {
    features <- compute_rff_cpp(Y.transformed, frequencies, num.threads)
  }

  result <- list(
    features = features,
    frequencies = frequencies,
    kernel = kernel,
    bandwidth = as.numeric(bandwidth),
    num.rff = num.rff,
    seed = as.integer(seed),
    backend = selected.backend,
    response.dimension = response.dimension,
    response.transform = transformed$name,
    response.transform.signature = transformed$signature,
    response.center = transformed$center,
    response.scale = transformed$scale,
    response.fingerprint = response_fingerprint(Y),
    transformed.response.fingerprint = response_fingerprint(Y.transformed),
    bytes = as.numeric(object.size(features) + object.size(frequencies))
  )
  class(result) <- "sdrf_rff"

  if (!is.null(output.file)) {
    output.dir <- dirname(output.file)
    if (!dir.exists(output.dir)) {
      dir.create(output.dir, recursive = TRUE, showWarnings = FALSE)
    }
    saveRDS(result, output.file, compress = FALSE)
  }
  result
}

#' Precompute RFF projections for rolling or expanding windows
#'
#' @param Y Full numeric response matrix ordered in time.
#' @param windows A list of integer training-row vectors, or a two-column matrix
#'   whose rows are inclusive start/end indices.
#' @param output.dir Directory for one uncompressed \code{.rds} file per
#'   window. On Hypatia this should normally be under \code{/sharedscratch}.
#' @param prefix File-name prefix.
#' @param overwrite Whether existing projection files may be replaced.
#' @inheritParams sdrf_rff
#'
#' @return A manifest data frame. Projection matrices are not retained in the
#'   returned object, preventing all windows from occupying RAM simultaneously.
#' @export
sdrf_rff_windows <- function(Y,
                             windows,
                             output.dir,
                             prefix = "sdrf-rff",
                             overwrite = FALSE,
                             num.rff = 256L,
                             kernel = c("gaussian", "laplacian"),
                             bandwidth = NULL,
                             response.transform = c("scale", "none"),
                             seed = 1L,
                             backend = c("auto", "cpu", "torch"),
                             num.threads = NULL) {
  if (missing(output.dir) || !nzchar(output.dir)) {
    stop("output.dir is required for windowed projection.")
  }
  kernel <- match.arg(kernel)
  backend <- match.arg(backend)
  window.indices <- normalize_rff_windows(windows, NROW(Y))
  dir.create(output.dir, recursive = TRUE, showWarnings = FALSE)

  manifest <- vector("list", length(window.indices))
  for (window in seq_along(window.indices)) {
    indices <- window.indices[[window]]
    output.file <- file.path(
      output.dir,
      sprintf("%s-%04d.rds", prefix, window)
    )
    window.response <- Y[indices, , drop = FALSE]
    if (file.exists(output.file) && !overwrite) {
      projection <- readRDS(output.file)
      validate_cached_rff(
        projection = projection,
        Y = window.response,
        num.rff = num.rff,
        kernel = kernel,
        bandwidth = bandwidth,
        response.transform = response.transform,
        seed = seed
      )
    } else {
      projection <- sdrf_rff(
        Y = window.response,
        num.rff = num.rff,
        kernel = kernel,
        bandwidth = bandwidth,
        response.transform = response.transform,
        seed = seed,
        backend = backend,
        num.threads = num.threads,
        output.file = output.file
      )
    }
    manifest[[window]] <- data.frame(
      window = window,
      start = min(indices),
      end = max(indices),
      observations = length(indices),
      kernel = projection$kernel,
      bandwidth = projection$bandwidth,
      num.rff = projection$num.rff,
      bytes = projection$bytes,
      path = normalizePath(output.file, winslash = "/", mustWork = FALSE),
      stringsAsFactors = FALSE
    )
    rm(projection)
  }

  manifest <- do.call(rbind, manifest)
  saveRDS(
    manifest,
    file.path(output.dir, paste0(prefix, "-manifest.rds")),
    compress = FALSE
  )
  manifest
}

transform_rff_response <- function(Y, response.transform) {
  if (is.function(response.transform)) {
    rows <- lapply(seq_len(NROW(Y)), function(row) {
      value <- response.transform(Y[row, ])
      if (!is.numeric(value) || anyNA(value) || any(!is.finite(value))) {
        stop("response.transform must return a finite numeric vector.")
      }
      as.numeric(value)
    })
    lengths <- vapply(rows, length, integer(1L))
    if (length(unique(lengths)) != 1L || lengths[1L] == 0L) {
      stop("response.transform must return a fixed, non-zero dimension.")
    }
    return(list(
      values = do.call(rbind, rows),
      name = "function",
      signature = paste(deparse(response.transform), collapse = "\n"),
      center = NULL,
      scale = NULL
    ))
  }

  response.transform <- match.arg(response.transform, c("scale", "none"))
  if (response.transform == "none") {
    return(list(values = Y, name = "none", signature = "none",
                center = NULL, scale = NULL))
  }

  scaled <- scale(Y)
  if (any(!is.finite(scaled))) {
    stop(
      "response.transform='scale' cannot scale a constant response column; ",
      "remove it, transform it explicitly, or use response.transform='none'."
    )
  }
  list(
    values = unclass(scaled),
    name = "scale",
    signature = "scale",
    center = attr(scaled, "scaled:center"),
    scale = attr(scaled, "scaled:scale")
  )
}

estimate_rff_bandwidth <- function(Y, kernel, max.rows = 2500L) {
  if (NROW(Y) > max.rows) {
    # Evenly spaced deterministic rows keep an automatically selected
    # bandwidth reproducible across preprocessing jobs and cache validation.
    rows <- unique(as.integer(round(seq.int(
      1L, NROW(Y), length.out = max.rows
    ))))
    Y <- Y[rows, , drop = FALSE]
  }
  method <- if (kernel == "gaussian") "euclidean" else "manhattan"
  bandwidth <- stats::median(as.numeric(stats::dist(Y, method = method)))
  if (!is.finite(bandwidth) || bandwidth <= 0) {
    stop("The median-distance bandwidth is zero; provide bandwidth explicitly.")
  }
  bandwidth
}

select_rff_backend <- function(backend, work) {
  torch.available <- function() {
    requireNamespace("torch", quietly = TRUE) &&
      isTRUE(tryCatch(torch::cuda_is_available(), error = function(error) FALSE))
  }
  if (backend == "torch" && !torch.available()) {
    stop(
      "backend='torch' requires the optional torch package with CUDA support."
    )
  }
  if (backend == "auto") {
    # GPU launch, transfer, and host-copy costs dominate small bivariate maps.
    # The threshold is conservative and should be benchmarked for each cluster.
    return(if (work >= 5e7 && torch.available()) "torch" else "cpu")
  }
  backend
}

project_rff_torch <- function(Y, frequencies) {
  response.tensor <- torch::torch_tensor(
    unname(Y), dtype = torch::torch_float64(), device = "cuda"
  )
  frequency.tensor <- torch::torch_tensor(
    unname(frequencies), dtype = torch::torch_float64(), device = "cuda"
  )
  phase <- response.tensor$matmul(frequency.tensor$t())
  scale <- sqrt(NROW(frequencies))
  real <- as.matrix(as.array((phase$cos() / scale)$cpu()))
  imaginary <- as.matrix(as.array((phase$sin() / scale)$cpu()))

  features <- matrix(0, nrow = NROW(Y), ncol = 2L * NROW(frequencies))
  features[, seq.int(1L, NCOL(features), by = 2L)] <- real
  features[, seq.int(2L, NCOL(features), by = 2L)] <- imaginary
  features
}

response_fingerprint <- function(Y) {
  path <- tempfile("sdrf-response-", fileext = ".rds")
  on.exit(unlink(path), add = TRUE)
  # MD5 is used only as an accidental-mismatch guard, not for security. Base
  # R's XDR serialization makes row order, dimensions, and every value part of
  # the digest without adding an external package dependency.
  saveRDS(unname(as.matrix(Y)), path, compress = FALSE, version = 2)
  unname(tools::md5sum(path))
}

validate_cached_rff <- function(projection,
                                Y,
                                num.rff,
                                kernel,
                                bandwidth,
                                response.transform,
                                seed) {
  if (!inherits(projection, "sdrf_rff") ||
      !identical(projection$response.fingerprint, response_fingerprint(Y))) {
    stop(
      "Existing RFF cache does not match this response window; use overwrite=TRUE."
    )
  }

  transformed <- transform_rff_response(as.matrix(Y), response.transform)
  expected.bandwidth <- bandwidth
  if (is.null(expected.bandwidth)) {
    expected.bandwidth <- estimate_rff_bandwidth(transformed$values, kernel)
  }
  metadata.matches <- identical(as.integer(projection$num.rff),
                                as.integer(num.rff)) &&
    identical(projection$kernel, kernel) &&
    isTRUE(all.equal(projection$bandwidth, as.numeric(expected.bandwidth),
                     tolerance = 1e-12)) &&
    identical(projection$response.transform.signature,
              transformed$signature) &&
    identical(projection$transformed.response.fingerprint,
          response_fingerprint(transformed$values)) &&
    identical(as.integer(projection$seed), as.integer(seed))
  if (!metadata.matches) {
    stop(
      "Existing RFF cache was created with different projection options; ",
      "use overwrite=TRUE."
    )
  }
  invisible(projection)
}

normalize_rff_windows <- function(windows, num.rows) {
  if (is.matrix(windows) && NCOL(windows) == 2L) {
    windows <- lapply(seq_len(NROW(windows)), function(row) {
      seq.int(windows[row, 1L], windows[row, 2L])
    })
  }
  if (!is.list(windows) || length(windows) == 0L) {
    stop("windows must be a non-empty list or a two-column matrix.")
  }
  lapply(windows, function(indices) {
    if (!is.numeric(indices) || length(indices) == 0L ||
        anyNA(indices) || any(indices != floor(indices)) ||
        any(indices < 1L | indices > num.rows) || anyDuplicated(indices)) {
      stop("Each window must contain unique, valid integer row indices.")
    }
    as.integer(indices)
  })
}