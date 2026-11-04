#!/usr/bin/env Rscript
# =============================================================================
# GAM EOF Interpolation for RPCA Wind Reconstruction (FAST VERSION)
# GeoSphere Austria - Wind Atlas Project
# =============================================================================
#
# Uses mgcv::bam() instead of BAMLSS for MUCH faster fitting.
# bam() is optimized for large datasets and uses deterministic optimization
# instead of MCMC sampling.
#
# Speed comparison:
#   - BAMLSS (2000 iter): ~30-60 min/EOF → 48-96 hours total
#   - GAM/bam():          ~10-30 sec/EOF → 15-45 min total
#
# Input:  rpca_export/year_YYYY/eof_coefficients_training.csv
# Output: output/eof_gam/eof_grids_gam_YYYY.nc
#
# Usage:
#   Rscript gam_rpca_eof.R --rpca_dir rpca_export/year_2020 --n_eofs 96
#
# =============================================================================

suppressPackageStartupMessages({
  library(optparse)
})

# -----------------------------------------------------------------------------
# COMMAND LINE ARGUMENTS
# -----------------------------------------------------------------------------

option_list <- list(
  make_option("--rpca_dir", type="character", default="rpca_export/year_2020",
              help="RPCA export directory [default %default]"),
  make_option("--covariate_dir", type="character", default="../DEM_DERIVATIVES/COVARIATES/1KM/",
              help="Covariate GeoTIFFs directory [default %default]"),
  make_option("--output_dir", type="character", default="output/eof_gam",
              help="Output directory [default %default]"),
  make_option("--n_eofs", type="integer", default=96,
              help="Number of EOFs to interpolate [default %default]"),
  make_option("--method", type="character", default="bam",
              help="Fitting method: bam (fast) or gam (standard) [default %default]"),
  make_option("--cores", type="integer", default=4,
              help="Parallel cores for bam [default %default]"),
  make_option("--quick", action="store_true", default=FALSE,
              help="Quick test mode (10 EOFs)")
)

opt <- parse_args(OptionParser(option_list=option_list))

if (opt$quick) {
  opt$n_eofs <- min(opt$n_eofs, 10)
  message("*** QUICK MODE: n_eofs=", opt$n_eofs, " ***")
}

# -----------------------------------------------------------------------------
# LOAD PACKAGES
# -----------------------------------------------------------------------------

message("\n[1/7] Loading packages...")

required_packages <- c("mgcv", "terra", "ncdf4", "parallel")

for (pkg in required_packages) {
  if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
    install.packages(pkg, repos = "https://cloud.r-project.org")
    library(pkg, character.only = TRUE)
  }
}

# Check for sf
has_sf <- require("sf", quietly = TRUE)

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------

# INCA Grid (EPSG:31287)
GRID_XMIN <- 19500;  GRID_XMAX <- 720500
GRID_YMIN <- 219500; GRID_YMAX <- 620500
GRID_NCOLS <- 701;   GRID_NROWS <- 401

x_coords <- seq(GRID_XMIN, GRID_XMAX, length.out = GRID_NCOLS)
y_coords <- seq(GRID_YMIN, GRID_YMAX, length.out = GRID_NROWS)

target_year <- as.integer(gsub(".*year_", "", opt$rpca_dir))

message("\n============================================================")
message("GAM EOF INTERPOLATION (FAST)")
message("============================================================")
message(sprintf("RPCA Dir:     %s", opt$rpca_dir))
message(sprintf("Target Year:  %d", target_year))
message(sprintf("N EOFs:       %d", opt$n_eofs))
message(sprintf("Method:       %s", opt$method))
message(sprintf("Cores:        %d", opt$cores))
message("============================================================\n")

# -----------------------------------------------------------------------------
# LOAD RPCA DATA
# -----------------------------------------------------------------------------

message("[2/7] Loading RPCA data...")

eof_file <- file.path(opt$rpca_dir, "eof_coefficients_training.csv")
if (!file.exists(eof_file)) stop("EOF file not found: ", eof_file)
eof_df <- read.csv(eof_file)
message(sprintf("  EOF coefficients: %d stations × %d columns", nrow(eof_df), ncol(eof_df)))

meta_file <- file.path(opt$rpca_dir, "station_metadata.csv")
if (!file.exists(meta_file)) stop("Metadata file not found: ", meta_file)
meta_df <- read.csv(meta_file)
message(sprintf("  Station metadata: %d stations", nrow(meta_df)))
message(sprintf("  Columns: %s", paste(names(meta_df), collapse=", ")))

mean_file <- file.path(opt$rpca_dir, "spatial_mean_field.csv")
if (!file.exists(mean_file)) stop("Spatial mean file not found: ", mean_file)
mean_df <- read.csv(mean_file)
message(sprintf("  Spatial mean: %d stations", nrow(mean_df)))
message(sprintf("  Columns: %s", paste(names(mean_df), collapse=", ")))

# Find EOF columns
eof_cols <- grep("^EOF_|^eof_|^basis_|^V\\d|^comp_|^PC", names(eof_df), value = TRUE)
if (length(eof_cols) == 0) {
  numeric_cols <- names(eof_df)[sapply(eof_df, is.numeric)]
  id_candidates <- c("w_id", "station_id", "id", "W_ID", "X", "Unnamed")
  eof_cols <- setdiff(numeric_cols, id_candidates)
}
message(sprintf("  Found %d EOF columns", length(eof_cols)))

n_available <- length(eof_cols)
if (opt$n_eofs > n_available) {
  message(sprintf("  Limiting to %d available EOFs", n_available))
  opt$n_eofs <- n_available
}
eof_cols <- eof_cols[1:opt$n_eofs]

# -----------------------------------------------------------------------------
# HELPER FUNCTION
# -----------------------------------------------------------------------------

find_column <- function(df, candidates) {
  for (cand in candidates) {
    matches <- grep(paste0("^", cand, "$"), names(df), ignore.case = TRUE, value = TRUE)
    if (length(matches) > 0) return(matches[1])
  }
  return(NULL)
}

# -----------------------------------------------------------------------------
# COORDINATES
# -----------------------------------------------------------------------------

# Find ID columns
id_col_meta <- find_column(meta_df, c("w_id", "station_id", "id"))
id_col_eof <- find_column(eof_df, c("w_id", "station_id", "id"))
id_col_mean <- find_column(mean_df, c("w_id", "station_id", "id"))

if (is.null(id_col_meta)) id_col_meta <- names(meta_df)[1]
if (is.null(id_col_eof)) id_col_eof <- names(eof_df)[1]
if (is.null(id_col_mean)) id_col_mean <- names(mean_df)[1]

message(sprintf("  ID columns: meta=%s, eof=%s, mean=%s", id_col_meta, id_col_eof, id_col_mean))

# Find coordinate columns
x_col <- find_column(meta_df, c("x", "X", "easting"))
y_col <- find_column(meta_df, c("y", "Y", "northing"))
lon_col <- find_column(meta_df, c("lon", "longitude"))
lat_col <- find_column(meta_df, c("lat", "latitude"))

# Check if x/y are Lambert coordinates
if (!is.null(x_col) && !is.null(y_col)) {
  x_range <- range(meta_df[[x_col]], na.rm = TRUE)
  if (x_range[1] > 1000 && x_range[2] > 10000) {
    message("  Using existing Lambert coordinates")
    meta_df$x_lambert <- meta_df[[x_col]]
    meta_df$y_lambert <- meta_df[[y_col]]
  } else if (!is.null(lon_col) && !is.null(lat_col)) {
    message("  Transforming lon/lat to Lambert...")
    if (has_sf) {
      pts <- st_as_sf(meta_df, coords = c(lon_col, lat_col), crs = 4326)
      pts_lambert <- st_transform(pts, 31287)
      coords <- st_coordinates(pts_lambert)
      meta_df$x_lambert <- coords[, 1]
      meta_df$y_lambert <- coords[, 2]
    } else {
      # Approximate transformation
      lon_0 <- 13.33333; lat_0 <- 47.5
      meta_df$x_lambert <- 400000 + (meta_df[[lon_col]] - lon_0) * 111320 * cos(lat_0 * pi/180)
      meta_df$y_lambert <- 400000 + (meta_df[[lat_col]] - lat_0) * 110540
    }
  }
} else if (!is.null(lon_col) && !is.null(lat_col)) {
  message("  Transforming lon/lat to Lambert...")
  if (has_sf) {
    pts <- st_as_sf(meta_df, coords = c(lon_col, lat_col), crs = 4326)
    pts_lambert <- st_transform(pts, 31287)
    coords <- st_coordinates(pts_lambert)
    meta_df$x_lambert <- coords[, 1]
    meta_df$y_lambert <- coords[, 2]
  } else {
    lon_0 <- 13.33333; lat_0 <- 47.5
    meta_df$x_lambert <- 400000 + (meta_df[[lon_col]] - lon_0) * 111320 * cos(lat_0 * pi/180)
    meta_df$y_lambert <- 400000 + (meta_df[[lat_col]] - lat_0) * 110540
  }
}

message(sprintf("  X range: [%.0f, %.0f]", min(meta_df$x_lambert), max(meta_df$x_lambert)))
message(sprintf("  Y range: [%.0f, %.0f]", min(meta_df$y_lambert), max(meta_df$y_lambert)))

# -----------------------------------------------------------------------------
# FIND SPATIAL MEAN COLUMN (FIXED)
# -----------------------------------------------------------------------------

# Look for column with 'mean' or 'spatial' in the name, but NOT coordinates
mean_col <- NULL
for (cand in c("spatial_mean", "mean_wind", "ff_mean", "wind_mean", "mean")) {
  matches <- grep(cand, names(mean_df), ignore.case = TRUE, value = TRUE)
  # Exclude coordinate-like columns
  matches <- matches[!grepl("^x$|^y$|^lon|^lat|^id", matches, ignore.case = TRUE)]
  if (length(matches) > 0) {
    mean_col <- matches[1]
    break
  }
}

# If still not found, take first numeric column that's not an ID or coordinate
if (is.null(mean_col)) {
  numeric_cols <- names(mean_df)[sapply(mean_df, is.numeric)]
  exclude <- c(id_col_mean, "x", "y", "X", "Y", "lon", "lat", "alt", "elevation")
  candidates <- setdiff(numeric_cols, exclude)
  if (length(candidates) > 0) {
    mean_col <- candidates[1]
  }
}

if (is.null(mean_col)) {
  stop("Could not find spatial mean column in ", mean_file)
}
message(sprintf("  Spatial mean column: %s", mean_col))

# -----------------------------------------------------------------------------
# MERGE DATA
# -----------------------------------------------------------------------------

station_data <- merge(meta_df, eof_df, by.x = id_col_meta, by.y = id_col_eof)
station_data <- merge(station_data, mean_df[, c(id_col_mean, mean_col)], 
                      by.x = id_col_meta, by.y = id_col_mean, all.x = TRUE)
names(station_data)[names(station_data) == mean_col] <- "spatial_mean"

message(sprintf("  Merged: %d stations", nrow(station_data)))

# -----------------------------------------------------------------------------
# LOAD COVARIATES
# -----------------------------------------------------------------------------

message("\n[3/7] Loading terrain covariates...")

covariate_patterns <- list(
  elevation = "EUDEM.*\\.tif$|DEM.*\\.tif$",
  tpi = "TPI.*\\.tif$",
  slope = "SLOPE.*\\.tif$",
  aspect_sin = "ASPECT_SIN.*\\.tif$",
  aspect_cos = "ASPECT_COS.*\\.tif$",
  roughness = "ROUGHNESS.*\\.tif$"
)

cov_rasters <- list()
for (cov_name in names(covariate_patterns)) {
  files <- list.files(opt$covariate_dir, pattern = covariate_patterns[[cov_name]], 
                      full.names = TRUE, ignore.case = TRUE)
  if (length(files) > 0) {
    cov_rasters[[cov_name]] <- terra::rast(files[1])
    message(sprintf("  %s: %s", cov_name, basename(files[1])))
  }
}

if (length(cov_rasters) == 0) stop("No covariates found!")
message(sprintf("  Loaded %d covariates", length(cov_rasters)))

# Extract at stations
message("  Extracting at stations...")
station_pts <- terra::vect(cbind(station_data$x_lambert, station_data$y_lambert), crs = "EPSG:31287")

for (cov_name in names(cov_rasters)) {
  extracted <- terra::extract(cov_rasters[[cov_name]], station_pts)
  station_data[[cov_name]] <- extracted[[2]]
}

# Remove NA
complete <- complete.cases(station_data[, names(cov_rasters)])
if (sum(!complete) > 0) {
  message(sprintf("  Removing %d stations with NA", sum(!complete)))
  station_data <- station_data[complete, ]
}
message(sprintf("  Training set: %d stations", nrow(station_data)))

# -----------------------------------------------------------------------------
# CREATE GRID
# -----------------------------------------------------------------------------

message("\n[4/7] Creating prediction grid...")

grid_df <- expand.grid(x = x_coords, y = y_coords)
message("  Extracting at grid points...")

grid_pts <- terra::vect(as.matrix(grid_df), crs = "EPSG:31287")
for (cov_name in names(cov_rasters)) {
  extracted <- terra::extract(cov_rasters[[cov_name]], grid_pts)
  grid_df[[cov_name]] <- extracted[[2]]
  # Fill NA
  na_idx <- is.na(grid_df[[cov_name]])
  if (any(na_idx)) {
    grid_df[[cov_name]][na_idx] <- mean(grid_df[[cov_name]], na.rm = TRUE)
  }
}

message(sprintf("  Grid: %d points", nrow(grid_df)))

# -----------------------------------------------------------------------------
# GAM FORMULA
# -----------------------------------------------------------------------------

cov_terms <- paste(sapply(names(cov_rasters), function(x) sprintf("s(%s, k=10)", x)), 
                   collapse = " + ")
formula_str <- sprintf("z ~ %s + s(x, y, k=30, bs='tp')", cov_terms)
gam_formula <- as.formula(formula_str)

message(sprintf("\n  Formula: %s", formula_str))

# -----------------------------------------------------------------------------
# FIT GAM FOR EACH EOF
# -----------------------------------------------------------------------------

message("\n[5/7] Fitting GAM models...")
message("============================================================\n")

# Setup parallel cluster for bam
if (opt$method == "bam" && opt$cores > 1) {
  message(sprintf("  Using %d cores for parallel fitting", opt$cores))
  cl <- makeCluster(opt$cores)
} else {
  cl <- NULL
}

eof_grids <- array(NA, dim = c(opt$n_eofs, GRID_NROWS, GRID_NCOLS))
fit_stats <- data.frame(eof = integer(), rmse = numeric(), cor = numeric(), time_sec = numeric())

total_start <- Sys.time()

for (i in 1:opt$n_eofs) {
  eof_col <- eof_cols[i]
  message(sprintf("EOF %d/%d: %s", i, opt$n_eofs, eof_col))
  
  start_time <- Sys.time()
  
  # Training data
  train_df <- data.frame(
    x = station_data$x_lambert,
    y = station_data$y_lambert,
    z = station_data[[eof_col]]
  )
  for (cov_name in names(cov_rasters)) {
    train_df[[cov_name]] <- station_data[[cov_name]]
  }
  train_df <- train_df[complete.cases(train_df), ]
  
  tryCatch({
    # Fit GAM/BAM
    if (opt$method == "bam") {
      fit <- bam(gam_formula, data = train_df, method = "fREML",
                 discrete = TRUE, nthreads = opt$cores)
    } else {
      fit <- gam(gam_formula, data = train_df, method = "REML")
    }
    
    # In-sample metrics
    pred_train <- predict(fit, newdata = train_df)
    rmse <- sqrt(mean((train_df$z - pred_train)^2))
    cor_val <- cor(train_df$z, pred_train)
    
    # Predict on grid
    pred_df <- data.frame(x = grid_df$x, y = grid_df$y)
    for (cov_name in names(cov_rasters)) {
      pred_df[[cov_name]] <- grid_df[[cov_name]]
    }
    
    grid_pred <- predict(fit, newdata = pred_df)
    
    # Reshape
    pred_matrix <- matrix(grid_pred, nrow = GRID_NCOLS, ncol = GRID_NROWS)
    eof_grids[i, , ] <- t(pred_matrix)
    
    elapsed <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
    fit_stats <- rbind(fit_stats, data.frame(eof = i, rmse = rmse, cor = cor_val, time_sec = elapsed))
    
    message(sprintf("  RMSE=%.4f, r=%.3f, time=%.1fs", rmse, cor_val, elapsed))
    
  }, error = function(e) {
    message(sprintf("  ERROR: %s", e$message))
  })
}

if (!is.null(cl)) stopCluster(cl)

# -----------------------------------------------------------------------------
# SPATIAL MEAN
# -----------------------------------------------------------------------------

message("\n[6/7] Interpolating spatial mean...")

mean_train_df <- data.frame(
  x = station_data$x_lambert,
  y = station_data$y_lambert,
  z = station_data$spatial_mean
)
for (cov_name in names(cov_rasters)) {
  mean_train_df[[cov_name]] <- station_data[[cov_name]]
}
mean_train_df <- mean_train_df[complete.cases(mean_train_df), ]

tryCatch({
  if (opt$method == "bam") {
    mean_fit <- bam(gam_formula, data = mean_train_df, method = "fREML", discrete = TRUE)
  } else {
    mean_fit <- gam(gam_formula, data = mean_train_df, method = "REML")
  }
  
  pred_df <- data.frame(x = grid_df$x, y = grid_df$y)
  for (cov_name in names(cov_rasters)) {
    pred_df[[cov_name]] <- grid_df[[cov_name]]
  }
  
  mean_pred <- predict(mean_fit, newdata = pred_df)
  mean_grid <- matrix(mean_pred, nrow = GRID_NCOLS, ncol = GRID_NROWS)
  mean_grid <- t(mean_grid)
  
  message("  Done")
}, error = function(e) {
  message(sprintf("  ERROR: %s - using constant mean", e$message))
  mean_grid <- matrix(mean(station_data$spatial_mean, na.rm = TRUE), GRID_NROWS, GRID_NCOLS)
})

# -----------------------------------------------------------------------------
# SAVE
# -----------------------------------------------------------------------------

message("\n[7/7] Saving NetCDF...")

dir.create(opt$output_dir, recursive = TRUE, showWarnings = FALSE)
output_file <- file.path(opt$output_dir, sprintf("eof_grids_gam_%d.nc", target_year))

x_dim <- ncdim_def("x", "m", x_coords)
y_dim <- ncdim_def("y", "m", y_coords)
eof_dim <- ncdim_def("eof", "", 1:opt$n_eofs)

eof_var <- ncvar_def("eof_coefficients", "1", list(eof_dim, y_dim, x_dim), missval = -9999)
mean_var <- ncvar_def("spatial_mean", "m/s", list(y_dim, x_dim), missval = -9999)

nc <- nc_create(output_file, list(eof_var, mean_var))
ncvar_put(nc, eof_var, eof_grids)
ncvar_put(nc, mean_var, mean_grid)

ncatt_put(nc, 0, "title", "GAM EOF Interpolation")
ncatt_put(nc, 0, "method", opt$method)
ncatt_put(nc, 0, "target_year", target_year)
ncatt_put(nc, 0, "n_eofs", opt$n_eofs)
ncatt_put(nc, 0, "n_stations", nrow(station_data))
ncatt_put(nc, 0, "crs", "EPSG:31287")

nc_close(nc)

stats_file <- file.path(opt$output_dir, sprintf("gam_fit_stats_%d.csv", target_year))
write.csv(fit_stats, stats_file, row.names = FALSE)

total_time <- as.numeric(difftime(Sys.time(), total_start, units = "mins"))

message(sprintf("\n  Output: %s", output_file))
message(sprintf("  Stats:  %s", stats_file))

# -----------------------------------------------------------------------------
# SUMMARY
# -----------------------------------------------------------------------------

message("\n============================================================")
message("COMPLETE")
message("============================================================")
message(sprintf("EOFs:            %d", nrow(fit_stats)))
message(sprintf("Mean RMSE:       %.4f", mean(fit_stats$rmse)))
message(sprintf("Mean correlation: %.3f", mean(fit_stats$cor)))
message(sprintf("Total time:      %.1f min", total_time))
message("============================================================")
message("\nNext: Run reconstruction:")
message(sprintf("  python rpca_reconstruct.py --eof_grids %s --method_name gam --monthly", output_file))
