library(reticulate)
use_condaenv("torch-openreml")

openreml <- import("torch_openreml", convert = FALSE)
torch <- import("torch", convert = FALSE)

BlockDiagonal <- openreml$covariance$BlockDiagonal
ScalarMatrix <- openreml$covariance$ScalarMatrix
IdentityMatrix <- openreml$covariance$IdentityMatrix
KroneckerProduct <- openreml$covariance$KroneckerProduct
CovariancePropagation <- openreml$covariance$CovariancePropagation
Sum <- openreml$covariance$Sum
MarginalREML <- openreml$MarginalREML

data <- agridat::john.alpha
n <- nrow(data)

y <- torch$tensor(data$yield, dtype = torch$float32)
X <- model.matrix(~ rep, data = data) |>
    torch$tensor(dtype = torch$float32)

Z <- model.matrix(~ 0 + gen + block:rep, data = data) |>
    torch$tensor(dtype = torch$float32)

V <- Sum(
    random = CovariancePropagation(
        Z = Z,
        G = BlockDiagonal(
            G_gen = ScalarMatrix(data$gen),
            G_rep_block = KroneckerProduct(
                G_rep = IdentityMatrix(data$rep),
                G_block = ScalarMatrix(data$block)
            )
        )
    ),
    residual = ScalarMatrix(n)
)

fit_openreml <- MarginalREML(V)
result <- fit_openreml$optimize(y, X, torch$zeros(3L), verbose = 2L)

print(py_to_r(fit_openreml$get_theta()$numpy()))
print(py_to_r(V$build_params(fit_openreml$get_theta())$numpy()))
print(py_to_r(fit_openreml$get_beta()$numpy()))

tree <- V$tree(fit_openreml$get_theta())[[0]]

b_hat <- openreml$blup(y, X, tree[["random/Z"]], tree[["random/G"]], tree[["/"]])
print(py_to_r(b_hat$numpy()))

Z_gen <- model.matrix(~ 0 + gen, data = data) |>
    torch$tensor(dtype = torch$float32)
gen_blup <- openreml$blup(y, X, Z_gen, tree[["random/G/G_gen"]], tree[["/"]])
print(setNames(py_to_r(gen_blup$numpy()), levels(data$gen)))
