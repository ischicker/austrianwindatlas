#!/usr/bin/env Rscript
# =============================================================================
# BAMLSS EOF Interpolation for RPCA Wind Reconstruction
# GeoSphere Austria - Wind Atlas Project
# =============================================================================
#
# Interpolates RPCA EOF coefficients from station locations to INCA grid
# using Bayesian Additive Models for Location, Scale and Shape.
#
# Input:  rpca_export/year_YYYY/eof_coefficients_training.csv
# Output: output/eof_bamlss/eof_grids_bamlss_YYYY.nc
#
# Usage:
#   Rscript bamlss_rpca_eof.R --rpca_dir rpca_export/year_2020 --n_eofs 96
#   Rscript bamlss_rpca_eof.R --rpca_dir rpca_export/year_2020 --n_eofs 30 --quick
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
  make_option("--output_dir", type="character", default="output/eof_bamlss",
              help="Output directory [default %default]"),
  make_option("--n_eofs", type="integer", default=96,
              help="Number of EOFs to interpolate [default %default]"),
  make_option("--n_iter", type="integer", default=2000,
              help="MCMC iterations [default %default]"),
  make_option("--burnin", type="integer", default=500,
              help="MCMC burnin [default %default]"),
  make_option("--cores", type="integer", default=4,
              help="Parallel cores [default %default]"),
  make_option("--quick", action="store_true", default=FALSE,
              help="Quick test mode (500 iter, 10 EOFs)")
)

opt <- parse_args(OptionParser(option_list=option_list))

if (opt$quick) {
  opt$n_iter <- 500
  opt$burnin <- 100
  opt$n_eofs <- min(opt$n_eofs, 10)
  message("*** QUICK MODE: n_iter=500, n_eofs=", opt$n_eofs, " ***")
}

# -----------------------------------------------------------------------------
# LOAD PACKAGES
# -----------------------------------------------------------------------------

message("\n[1/7] Loading packages...")

required_packages <- c("bamlss", "terra", "mgcv", "ncdf4", "parallel")

for (pkg in required_packages) {
  if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
    install.packages(pkg, repos = "https://cloud.r-project.org")
    library(pkg, character.only = TRUE)
  }
}

# Check for sf (for coordinate transformation)
has_sf <- require("sf", quietly = TRUE)
if (!has_sf) {
  message("  Note: sf package not available, assuming coordinates are already in EPSG:31287")
}

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------

# INCA Grid (EPSG:31287)
GRID_XMIN <- 19500;  GRID_XMAX <- 720500
GRID_YMIN <- 219500; GRID_YMAX <- 620500
GRID_NCOLS <- 701;   GRID_NROWS <- 401

x_coords <- seq(GRID_XMIN, GRID_XMAX, length.out = GRID_NCOLS)
y_coords <- seq(GRID_YMIN, GRID_YMAX, length.out = GRID_NROWS)

# Extract year from rpca_dir
target_year <- as.integer(gsub(".*year_", "", opt$rpca_dir))

message("\n============================================================")
message("BAMLSS EOF INTERPOLATION")
message("============================================================")
message(sprintf("RPCA Dir:     %s", opt$rpca_dir))
message(sprintf("Target Year:  %d", target_year))
message(sprintf("N EOFs:       %d", opt$n_eofs))
message(sprintf("MCMC:         %d iter, %d burnin", opt$n_iter, opt$burnin))
message(sprintf("Cores:        %d", opt$cores))
message("============================================================\n")

# -----------------------------------------------------------------------------
# LOAD RPCA DATA
# -----------------------------------------------------------------------------

message("[2/7] Loading RPCA data...")

# EOF coefficients (V @ Sigma from SVD)
eof_file <- file.path(opt$rpca_dir, "eof_coefficients_training.csv")
if (!file.exists(eof_file)) stop("EOF file not found: ", eof_file)
eof_df <- read.csv(eof_file)
message(sprintf("  EOF coefficients: %d stations × %d columns", nrow(eof_df), ncol(eof_df)))

# Station metadata (coordinates)
meta_file <- file.path(opt$rpca_dir, "station_metadata.csv")
if (!file.exists(meta_file)) stop("Metadata file not found: ", meta_file)
meta_df <- read.csv(meta_file)
message(sprintf("  Station metadata: %d stations", nrow(meta_df)))
message(sprintf("  Metadata columns: %s", paste(names(meta_df), collapse=", ")))

# Spatial mean
mean_file <- file.path(opt$rpca_dir, "spatial_mean_field.csv")
if (!file.exists(mean_file)) stop("Spatial mean file not found: ", mean_file)
mean_df <- read.csv(mean_file)
message(sprintf("  Spatial mean: %d stations", nrow(mean_df)))

# Identify EOF columns (numeric columns that look like EOF coefficients)
eof_cols <- grep("^eof_|^basis_|^V\\d|^comp_|^PC", names(eof_df), value = TRUE)
if (length(eof_cols) == 0) {
  # Assume all numeric columns except first are EOFs
  numeric_cols <- names(eof_df)[sapply(eof_df, is.numeric)]
  # Remove ID column if present
  id_candidates <- c("w_id", "station_id", "id", "W_ID", "X")
  eof_cols <- setdiff(numeric_cols, id_candidates)
}
message(sprintf("  Found %d EOF columns", length(eof_cols)))

# Limit EOFs
n_available <- length(eof_cols)
if (opt$n_eofs > n_available) {
  message(sprintf("  Requested %d EOFs but only %d available", opt$n_eofs, n_available))
  opt$n_eofs <- n_available
}
eof_cols <- eof_cols[1:opt$n_eofs]

# -----------------------------------------------------------------------------
# IDENTIFY ID AND COORDINATE COLUMNS
# -----------------------------------------------------------------------------

# Find ID column
find_column <- function(df, candidates) {
  for (cand in candidates) {
    matches <- grep(paste0("^", cand, "$"), names(df), ignore.case = TRUE, value = TRUE)
    if (length(matches) > 0) return(matches[1])
  }
  return(NULL)
}

id_col_meta <- find_column(meta_df, c("w_id", "station_id", "id", "W_ID"))
id_col_eof <- find_column(eof_df, c("w_id", "station_id", "id", "W_ID"))
id_col_mean <- find_column(mean_df, c("w_id", "station_id", "id", "W_ID"))

if (is.null(id_col_meta)) id_col_meta <- names(meta_df)[1]
if (is.null(id_col_eof)) id_col_eof <- names(eof_df)[1]
if (is.null(id_col_mean)) id_col_mean <- names(mean_df)[1]

message(sprintf("  ID columns: meta=%s, eof=%s, mean=%s", id_col_meta, id_col_eof, id_col_mean))

# Find coordinate columns
x_col <- find_column(meta_df, c("x", "X", "easting", "EASTING"))
y_col <- find_column(meta_df, c("y", "Y", "northing", "NORTHING"))
lon_col <- find_column(meta_df, c("lon", "LON", "longitude", "LONGITUDE"))
lat_col <- find_column(meta_df, c("lat", "LAT", "latitude", "LATITUDE"))

message(sprintf("  Coordinate columns found: x=%s, y=%s, lon=%s, lat=%s", 
                ifelse(is.null(x_col), "NA", x_col),
                ifelse(is.null(y_col), "NA", y_col),
                ifelse(is.null(lon_col), "NA", lon_col),
                ifelse(is.null(lat_col), "NA", lat_col)))

# Determine if we need coordinate transformation
need_transform <- FALSE
if (!is.null(x_col) && !is.null(y_col)) {
  # Check if x/y look like Lambert coordinates (large values)
  x_range <- range(meta_df[[x_col]], na.rm = TRUE)
  if (x_range[1] > 1000 && x_range[2] > 10000) {
    message("  Using existing x/y columns (appear to be Lambert coordinates)")
    meta_df$x_lambert <- meta_df[[x_col]]
    meta_df$y_lambert <- meta_df[[y_col]]
  } else {
    need_transform <- TRUE
  }
} else if (!is.null(lon_col) && !is.null(lat_col)) {
  need_transform <- TRUE
} else {
  stop("Could not find coordinate columns in station metadata!")
}

# Transform coordinates if needed
if (need_transform) {
  message("  Transforming coordinates to EPSG:31287...")
  
  if (!is.null(lon_col) && !is.null(lat_col)) {
    lon <- meta_df[[lon_col]]
    lat <- meta_df[[lat_col]]
  } else {
    lon <- meta_df[[x_col]]
    lat <- meta_df[[y_col]]
  }
  
  if (has_sf) {
    # Use sf for proper transformation
    pts <- st_as_sf(data.frame(lon = lon, lat = lat), 
                    coords = c("lon", "lat"), crs = 4326)
    pts_lambert <- st_transform(pts, 31287)
    coords_lambert <- st_coordinates(pts_lambert)
    meta_df$x_lambert <- coords_lambert[, 1]
    meta_df$y_lambert <- coords_lambert[, 2]
  } else {
    # Approximate transformation (fallback)
    lon_0 <- 13.33333333
    lat_0 <- 47.5
    x_0 <- 400000
    y_0 <- 400000
    m_per_deg_lon <- 111320 * cos(lat_0 * pi / 180)
    m_per_deg_lat <- 110540
    meta_df$x_lambert <- x_0 + (lon - lon_0) * m_per_deg_lon
    meta_df$y_lambert <- y_0 + (lat - lat_0) * m_per_deg_lat
    message("  Warning: Using approximate transformation (install sf for exact)")
  }
  
  message(sprintf("  Lambert X range: [%.0f, %.0f]", 
                  min(meta_df$x_lambert), max(meta_df$x_lambert)))
  message(sprintf("  Lambert Y range: [%.0f, %.0f]", 
                  min(meta_df$y_lambert), max(meta_df$y_lambert)))
}

# -----------------------------------------------------------------------------
# MERGE DATA
# -----------------------------------------------------------------------------

# Get spatial mean column
mean_col <- find_column(mean_df, c("spatial_mean", "mean_wind", "ff_mean", "wind_mean"))
if (is.null(mean_col)) {
  numeric_cols <- names(mean_df)[sapply(mean_df, is.numeric)]
  exclude <- c(id_col_mean, "x", "y", "X", "Y", "lon", "lat", "alt", "elevation",
               "station_id", "w_id")
  candidates <- setdiff(numeric_cols, exclude)
  if (length(candidates) > 0) mean_col <- candidates[1]
}
message(sprintf("  Spatial mean column: %s", mean_col))

# Merge datasets
station_data <- merge(meta_df, eof_df, by.x = id_col_meta, by.y = id_col_eof)
station_data <- merge(station_data, mean_df[, c(id_col_mean, mean_col)], 
                      by.x = id_col_meta, by.y = id_col_mean, all.x = TRUE)

# Rename spatial mean column for consistency
names(station_data)[names(station_data) == mean_col] <- "spatial_mean"

message(sprintf("  Merged station data: %d stations", nrow(station_data)))

# -----------------------------------------------------------------------------
# LOAD COVARIATES
# -----------------------------------------------------------------------------

message("\n[3/7] Loading terrain covariates...")

covariate_patterns <- list(
  elevation = "EUDEM.*\\.tif$|DEM.*\\.tif$|elevation.*\\.tif$",
  tpi = "TPI.*\\.tif$",
  slope = "^SLOPE.*\\.tif$",
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
  } else {
    message(sprintf("  %s: NOT FOUND", cov_name))
  }
}

if (length(cov_rasters) == 0) {
  stop("No covariates found in ", opt$covariate_dir)
}
message(sprintf("  Loaded %d covariates", length(cov_rasters)))

# Extract at stations using proper terra::extract
message("  Extracting covariates at station locations...")

# Create SpatVector from station coordinates
station_pts <- terra::vect(
  cbind(station_data$x_lambert, station_data$y_lambert),
  crs = "EPSG:31287"
)

for (cov_name in names(cov_rasters)) {
  # terra::extract returns a data.frame with ID column and value column
  extracted <- terra::extract(cov_rasters[[cov_name]], station_pts)
  
  # The value column is typically the second column (after ID)
  # But let's be safe and get by position after ID
  if (ncol(extracted) >= 2) {
    vals <- extracted[[2]]  # Second column contains values
  } else {
    vals <- rep(NA, nrow(station_data))
  }
  
  station_data[[cov_name]] <- vals
  n_na <- sum(is.na(vals))
  val_range <- range(vals, na.rm = TRUE)
  message(sprintf("    %s: range [%.2f, %.2f], %d NA values", 
                  cov_name, val_range[1], val_range[2], n_na))
}

# Remove stations with NA covariates
complete <- complete.cases(station_data[, names(cov_rasters)])
if (sum(!complete) > 0) {
  message(sprintf("  Removing %d stations with NA covariates", sum(!complete)))
  station_data <- station_data[complete, ]
}

message(sprintf("  Training set: %d stations", nrow(station_data)))

# -----------------------------------------------------------------------------
# CREATE PREDICTION GRID
# -----------------------------------------------------------------------------

message("\n[4/7] Creating prediction grid...")

grid_df <- expand.grid(x = x_coords, y = y_coords)

# Create SpatVector for grid points
message("  Extracting covariates at grid points (this may take a moment)...")
grid_pts <- terra::vect(as.matrix(grid_df), crs = "EPSG:31287")

for (cov_name in names(cov_rasters)) {
  extracted <- terra::extract(cov_rasters[[cov_name]], grid_pts)
  if (ncol(extracted) >= 2) {
    vals <- extracted[[2]]
  } else {
    vals <- rep(NA, nrow(grid_df))
  }
  grid_df[[cov_name]] <- vals
  
  # Fill NA with mean for prediction (will be masked later)
  na_idx <- is.na(grid_df[[cov_name]])
  if (any(na_idx)) {
    grid_df[[cov_name]][na_idx] <- mean(grid_df[[cov_name]], na.rm = TRUE)
  }
}

n_grid_na <- sum(!complete.cases(grid_df[, names(cov_rasters)]))
message(sprintf("  Grid: %d × %d = %d points (%d outside Austria)", 
                GRID_NCOLS, GRID_NROWS, nrow(grid_df), n_grid_na))

# -----------------------------------------------------------------------------
# BAMLSS FORMULA
# -----------------------------------------------------------------------------

# Smooth terms for each covariate + spatial smooth
cov_terms <- paste(sapply(names(cov_rasters), function(x) sprintf("s(%s, k=10)", x)), 
                   collapse = " + ")
formula_str <- sprintf("z ~ %s + s(x, y, k=50)", cov_terms)
bamlss_formula <- as.formula(formula_str)

message(sprintf("\n  Formula: %s", formula_str))

# -----------------------------------------------------------------------------
# FIT BAMLSS FOR EACH EOF
# -----------------------------------------------------------------------------

message("\n[5/7] Fitting BAMLSS models...")
message("============================================================\n")

# Storage
eof_grids <- array(NA, dim = c(opt$n_eofs, GRID_NROWS, GRID_NCOLS))
fit_stats <- data.frame(eof = integer(), rmse = numeric(), cor = numeric(), time_sec = numeric())

for (i in 1:opt$n_eofs) {
  eof_col <- eof_cols[i]
  message(sprintf("EOF %d/%d: %s", i, opt$n_eofs, eof_col))
  
  start_time <- Sys.time()
  
  # Training data - use Lambert coordinates
  train_df <- data.frame(
    x = station_data$x_lambert,
    y = station_data$y_lambert
  )
  for (cov_name in names(cov_rasters)) {
    train_df[[cov_name]] <- station_data[[cov_name]]
  }
  train_df$z <- station_data[[eof_col]]
  
  # Remove any remaining NA
  train_df <- train_df[complete.cases(train_df), ]
  
  if (nrow(train_df) < 20) {
    message(sprintf("  WARNING: Only %d valid training points, using IDW fallback", nrow(train_df)))
    # IDW fallback for this EOF
    for (j in 1:nrow(grid_df)) {
      dx <- grid_df$x[j] - train_df$x
      dy <- grid_df$y[j] - train_df$y
      d <- sqrt(dx^2 + dy^2)
      d[d < 100] <- 100
      w <- 1 / d^2
      w <- w / sum(w)
      row_idx <- ((j - 1) %% GRID_NROWS) + 1
      col_idx <- ((j - 1) %/% GRID_NROWS) + 1
      eof_grids[i, row_idx, col_idx] <- sum(w * train_df$z)
    }
    next
  }
  
  tryCatch({
    # Fit BAMLSS
    fit <- bamlss(bamlss_formula, data = train_df, family = "gaussian",
                  n.iter = opt$n_iter, burnin = opt$burnin, thin = 1,
                  verbose = FALSE, cores = min(opt$cores, 2))
    
    # In-sample metrics
    pred_train <- predict(fit, newdata = train_df, type = "parameter")$mu
    rmse <- sqrt(mean((train_df$z - pred_train)^2))
    cor_val <- cor(train_df$z, pred_train)
    
    # Predict on grid
    # Rename grid columns to match training data
    pred_df <- data.frame(x = grid_df$x, y = grid_df$y)
    for (cov_name in names(cov_rasters)) {
      pred_df[[cov_name]] <- grid_df[[cov_name]]
    }
    
    grid_pred <- predict(fit, newdata = pred_df, type = "parameter")$mu
    
    # Reshape: expand.grid has x varying fastest, so matrix is (ncols, nrows)
    pred_matrix <- matrix(grid_pred, nrow = GRID_NCOLS, ncol = GRID_NROWS)
    pred_matrix <- t(pred_matrix)  # Transpose to (nrows, ncols)
    
    eof_grids[i, , ] <- pred_matrix
    
    elapsed <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
    fit_stats <- rbind(fit_stats, data.frame(eof = i, rmse = rmse, cor = cor_val, time_sec = elapsed))
    
    message(sprintf("  RMSE=%.4f, r=%.3f, time=%.1fs", rmse, cor_val, elapsed))
    
  }, error = function(e) {
    message(sprintf("  ERROR: %s - using IDW fallback", e$message))
    # IDW fallback
    for (j in 1:nrow(grid_df)) {
      dx <- grid_df$x[j] - train_df$x
      dy <- grid_df$y[j] - train_df$y
      d <- sqrt(dx^2 + dy^2)
      d[d < 100] <- 100
      w <- 1 / d^2
      w <- w / sum(w)
      row_idx <- ((j - 1) %% GRID_NROWS) + 1
      col_idx <- ((j - 1) %/% GRID_NROWS) + 1
      eof_grids[i, row_idx, col_idx] <- sum(w * train_df$z)
    }
  })
}

# -----------------------------------------------------------------------------
# INTERPOLATE SPATIAL MEAN
# -----------------------------------------------------------------------------

message("\n[6/7] Interpolating spatial mean...")

mean_train_df <- data.frame(
  x = station_data$x_lambert,
  y = station_data$y_lambert
)
for (cov_name in names(cov_rasters)) {
  mean_train_df[[cov_name]] <- station_data[[cov_name]]
}
mean_train_df$z <- station_data$spatial_mean
mean_train_df <- mean_train_df[complete.cases(mean_train_df), ]

tryCatch({
  mean_fit <- bamlss(bamlss_formula, data = mean_train_df, family = "gaussian",
                     n.iter = opt$n_iter, burnin = opt$burnin, verbose = FALSE, cores = 2)
  
  pred_df <- data.frame(x = grid_df$x, y = grid_df$y)
  for (cov_name in names(cov_rasters)) {
    pred_df[[cov_name]] <- grid_df[[cov_name]]
  }
  
  mean_pred <- predict(mean_fit, newdata = pred_df, type = "parameter")$mu
  mean_grid <- matrix(mean_pred, nrow = GRID_NCOLS, ncol = GRID_NROWS)
  mean_grid <- t(mean_grid)
  
  message("  Spatial mean interpolation complete")
  
}, error = function(e) {
  message(sprintf("  ERROR interpolating spatial mean: %s", e$message))
  message("  Using simple mean value...")
  mean_grid <- matrix(mean(station_data$spatial_mean, na.rm = TRUE), 
                      nrow = GRID_NROWS, ncol = GRID_NCOLS)
})

# -----------------------------------------------------------------------------
# SAVE NETCDF
# -----------------------------------------------------------------------------

message("\n[7/7] Saving NetCDF...")

dir.create(opt$output_dir, recursive = TRUE, showWarnings = FALSE)
output_file <- file.path(opt$output_dir, sprintf("eof_grids_bamlss_%d.nc", target_year))

# Create dimensions
x_dim <- ncdim_def("x", "m", x_coords)
y_dim <- ncdim_def("y", "m", y_coords)
eof_dim <- ncdim_def("eof", "", 1:opt$n_eofs)

# Create variables
eof_var <- ncvar_def("eof_coefficients", "1", list(eof_dim, y_dim, x_dim), missval = -9999)
mean_var <- ncvar_def("spatial_mean", "m/s", list(y_dim, x_dim), missval = -9999)

nc <- nc_create(output_file, list(eof_var, mean_var))
ncvar_put(nc, eof_var, eof_grids)
ncvar_put(nc, mean_var, mean_grid)

ncatt_put(nc, 0, "title", "BAMLSS EOF Interpolation")
ncatt_put(nc, 0, "method", "bamlss")
ncatt_put(nc, 0, "target_year", target_year)
ncatt_put(nc, 0, "n_eofs", opt$n_eofs)
ncatt_put(nc, 0, "n_stations", nrow(station_data))
ncatt_put(nc, 0, "crs", "EPSG:31287")
ncatt_put(nc, 0, "created", format(Sys.time(), "%Y-%m-%d %H:%M:%S"))

nc_close(nc)

# Save stats
stats_file <- file.path(opt$output_dir, sprintf("bamlss_fit_stats_%d.csv", target_year))
write.csv(fit_stats, stats_file, row.names = FALSE)

message(sprintf("\n  Output: %s", output_file))
message(sprintf("  Stats:  %s", stats_file))

# -----------------------------------------------------------------------------
# SUMMARY
# -----------------------------------------------------------------------------

message("\n============================================================")
message("COMPLETE")
message("============================================================")
message(sprintf("EOFs:            %d", nrow(fit_stats)))
if (nrow(fit_stats) > 0) {
  message(sprintf("Mean RMSE:       %.4f", mean(fit_stats$rmse)))
  message(sprintf("Mean correlation: %.3f", mean(fit_stats$cor)))
  message(sprintf("Total time:      %.1f min", sum(fit_stats$time_sec) / 60))
}
message("============================================================")
message("\nNext: Run reconstruction with rpca_reconstruct.py")
message(sprintf("  python rpca_reconstruct.py --eof_grids %s --monthly", output_file))
