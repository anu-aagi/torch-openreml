# Install R packages for this project -------------------------------------

install.packages(c("tidyverse", "reticulate",
                   "cli", "glue", "yardstick",
                   "remotes", "htmltools"))
remotes::install_github("TengMCing/scrubwren")


# Install Conda -----------------------------------------------------------

# Install `miniconda`
# Skip if `conda` exists in the system
if (is.null(reticulate:::find_conda()[[1]])) {
  reticulate::install_miniconda()
}

# You could use `options(reticulate.conda_binary = "/path/to/conda")` to
# force `reticulate` to use a particular `conda` binary

# Install Python libraries ------------------------------------------------

# Create an environment
if (reticulate::condaenv_exists("torch-openreml")) {
  reticulate::conda_remove("torch-openreml")
}

reticulate::conda_create("torch-openreml",
                         python_version = "3.12.12")

# Install libraries
reticulate::conda_install("torch-openreml",
                          pip = TRUE,
                          packages = c("-r", here::here("tests/R/requirements.txt")))
