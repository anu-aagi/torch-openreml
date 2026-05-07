library(reticulate)
use_condaenv("torch-openreml")

openreml <- import("torch_openreml", convert = FALSE)
torch <- import("torch", convert = FALSE)

ScalarMatrix <- openreml$covariance$ScalarMatrix
IdentityMatrix <- openreml$covariance$IdentityMatrix
KroneckerProduct <- openreml$covariance$KroneckerProduct
CovariancePropagation <- openreml$covariance$CovariancePropagation
Sum <- openreml$covariance$Sum
REML <- openreml$REML

data <- agridat::john.alpha

y <- torch$tensor(data$yield, dtype = torch$float32)
X <- model.matrix(~ rep, data = data) |>
    torch$tensor(dtype = torch$float32)

Z_gen <- model.matrix(~ gen - 1, data = data) |>
    torch$tensor(dtype = torch$float32)
Z_rep_block <- model.matrix(~ interaction(rep, block, lex.order = TRUE) - 1, data = data) |>
    torch$tensor(dtype = torch$float32)

G_gen <- ScalarMatrix(length(unique(data$gen)))
G_rep_block <- KroneckerProduct(IdentityMatrix(length(unique(data$rep))),
                                ScalarMatrix(length(unique(data$block))))
V <- Sum(CovariancePropagation(Z_gen, G_gen),
         CovariancePropagation(Z_rep_block, G_rep_block),
         ScalarMatrix(nrow(data)))

fit_openreml <- REML(V)
result <- fit_openreml$optimize(y, X, torch$zeros(3L), verbose = 2L)

print(py_to_r(fit_openreml$get_theta()$numpy()))
print(py_to_r(V$trans_params(fit_openreml$get_theta())$numpy()))
print(py_to_r(fit_openreml$get_beta()$numpy()))