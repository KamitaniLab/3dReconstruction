options(repos = c(CRAN = "https://cloud.r-project.org"))
Sys.setenv(MAKEFLAGS = "-j1")

install.packages("remotes")

install_exact <- function(package, version) {
  remotes::install_version(
    package,
    version = version,
    repos = "https://cloud.r-project.org",
    upgrade = "never",
    Ncpus = 1
  )
}

packages <- c(
  "lme4" = "1.1-37",
  "lmerTest" = "3.1-3",
  "pbkrtest" = "0.5.5",
  "broom" = "1.0.9",
  "broom.mixed" = "0.2.9.6",
  "emmeans" = "1.11.2-8"
)

for (package in names(packages)) {
  install_exact(package, packages[[package]])
}

expected_r <- "4.3.3"
actual_r <- paste(
  R.version$major,
  sub("^\\.", "", R.version$minor),
  sep = "."
)
if (actual_r != expected_r) {
  stop(sprintf("Expected R %s, got R %s", expected_r, actual_r))
}

for (package in names(packages)) {
  actual <- as.character(packageVersion(package))
  expected <- gsub("-", ".", packages[[package]], fixed = TRUE)
  if (actual != expected) {
    stop(sprintf("Expected %s %s, got %s", package, expected, actual))
  }
}

cat("R package versions match the LMM analysis targets.\n")
