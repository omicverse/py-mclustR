## Run R mclust on a battery of datasets/models and dump everything
## that py-mclustR's tests need to compare against. Uses jsonlite to
## emit a portable JSON record per (dataset, modelName, G); the
## responsibility matrix and parameters are flattened to nested numeric
## arrays so Python can rebuild them with numpy.

suppressPackageStartupMessages({
  library(mclust)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
out_dir <- if (length(args) >= 1) args[[1]] else "/scratch/users/steorra/analysis/omicverse_dev/py-mclustR/tests/_rparity"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

set.seed(20260417)
make_blobs <- function(n_per = 100, centers, sd = 0.7) {
  d <- ncol(centers)
  K <- nrow(centers)
  X <- matrix(0, n_per * K, d)
  y <- integer(n_per * K)
  for (k in seq_len(K)) {
    rows <- ((k - 1) * n_per + 1):(k * n_per)
    X[rows, ] <- matrix(rnorm(n_per * d, sd = sd), n_per, d) +
                  matrix(centers[k, ], n_per, d, byrow = TRUE)
    y[rows] <- k
  }
  list(X = X, y = y)
}

# Datasets ---------------------------------------------------------
data_blobs2 <- {
  centers <- matrix(c(0,0, 4,0, 2, 4), nrow = 3, byrow = TRUE)
  make_blobs(100, centers, sd = 0.7)
}
data_blobs5 <- {
  centers <- matrix(c(0,0,0,0,0,
                      6,0,0,0,0,
                      3,5,0,0,0,
                      0,0,5,0,0,
                      0,5,5,5,0), nrow = 5, byrow = TRUE)
  make_blobs(80, centers, sd = 1.0)
}
data_iris <- {
  d <- as.matrix(iris[, 1:4])
  list(X = d, y = as.integer(iris$Species))
}
data_faithful <- {
  d <- as.matrix(faithful)
  list(X = d, y = NULL)
}

datasets <- list(
  blobs2     = data_blobs2,
  blobs5     = data_blobs5,
  iris       = data_iris,
  faithful   = data_faithful
)

models <- c("EII", "VII", "EEI", "VEI", "EVI", "VVI",
            "EEE", "VEE", "EVE", "VVE",
            "EEV", "VEV", "EVV", "VVV")

# helper — collapse a list of params to JSON-friendly form
me_to_json <- function(ans) {
  if (!is.null(ans$error)) return(list(error = ans$error))
  pars <- ans$parameters
  variance <- pars$variance
  sigma <- variance$sigma  # (d, d, G)
  if (is.null(sigma)) {
    if (!is.null(variance$cholsigma)) {
      sigma <- variance$cholsigma
    } else if (!is.null(variance$Sigma)) {
      sigma <- variance$Sigma
    }
  }
  list(
    modelName = ans$modelName,
    n = ans$n,
    d = ans$d,
    G = ans$G,
    loglik = ans$loglik,
    pro = as.numeric(pars$pro),
    mean = as.numeric(pars$mean),       # column-major (d × G)
    mean_dim = dim(as.matrix(pars$mean)),
    sigma = if (!is.null(sigma)) as.numeric(sigma) else NULL,
    sigma_dim = if (!is.null(sigma)) dim(as.array(sigma)) else NULL,
    z = as.numeric(ans$z),
    z_dim = dim(as.matrix(ans$z))
  )
}

# Save data + per-record JSON ---------------------------------------
manifest <- list()

for (dname in names(datasets)) {
  ds <- datasets[[dname]]
  X <- as.matrix(ds$X)
  n <- nrow(X); d <- ncol(X)
  cat(sprintf("[%s] n=%d d=%d\n", dname, n, d))
  hc_obj <- hc(X, modelName = "VVV", use = "SVD")
  # Flat numeric dump for easy numpy ingest
  write.table(X, file.path(out_dir, paste0("data_", dname, "_X.csv")),
              sep = ",", row.names = FALSE, col.names = FALSE)
  if (!is.null(ds$y))
    write.table(matrix(ds$y), file.path(out_dir, paste0("data_", dname, "_y.csv")),
                sep = ",", row.names = FALSE, col.names = FALSE)

  for (G in 2:5) {
    cls <- hclass(hc_obj, G)
    z <- unmap(cls)
    write.table(z, file.path(out_dir, sprintf("zinit_%s_G%d.csv", dname, G)),
                sep = ",", row.names = FALSE, col.names = FALSE)
    for (mname in models) {
      key <- sprintf("%s_%s_G%d", dname, mname, G)
      ans <- tryCatch(
        me(modelName = mname, data = X, z = z),
        error = function(e) list(error = conditionMessage(e))
      )
      payload <- me_to_json(ans)
      payload$dataset <- dname
      payload$key <- key
      writeLines(toJSON(payload, auto_unbox = TRUE, digits = NA, na = "string"),
                 file.path(out_dir, paste0(key, ".json")))
      manifest[[length(manifest) + 1]] <- list(
        dataset = dname, modelName = mname, G = G,
        loglik = if (is.null(ans$loglik)) NA else ans$loglik,
        n = n, d = d, key = key
      )
    }
  }
  # Full Mclust(data, G=1:6) so we can compare model selection too
  m <- tryCatch(
    Mclust(X, G = 1:6, verbose = FALSE),
    error = function(e) list(error = conditionMessage(e))
  )
  if (is.null(m$error)) {
    full <- list(
      dataset = dname,
      modelName = m$modelName,
      G = m$G,
      loglik = m$loglik,
      bic = m$bic,
      df = m$df,
      classification = as.integer(m$classification),
      pro = as.numeric(m$parameters$pro),
      mean = as.numeric(m$parameters$mean),
      mean_dim = dim(as.matrix(m$parameters$mean)),
      sigma = as.numeric(m$parameters$variance$sigma),
      sigma_dim = dim(as.array(m$parameters$variance$sigma)),
      BIC = as.numeric(m$BIC),
      BIC_dimnames_G = rownames(m$BIC),
      BIC_dimnames_model = colnames(m$BIC),
      BIC_dim = dim(m$BIC)
    )
    writeLines(toJSON(full, auto_unbox = TRUE, digits = NA, na = "string"),
               file.path(out_dir, paste0("mclust_full_", dname, ".json")))
  }
}

writeLines(toJSON(manifest, pretty = TRUE, auto_unbox = TRUE, digits = NA, na = "string"),
           file.path(out_dir, "manifest.json"))
cat("DONE — wrote", length(manifest), "ME records\n")
