library(reticulate)
use_condaenv("torch-openreml")

openreml <- import("torch_openreml", convert = FALSE)
torch <- import("torch", convert = FALSE)

ScalarMatrix <- openreml$covariance$ScalarMatrix
IdentityMatrix <- openreml$covariance$IdentityMatrix
KroneckerProduct <- openreml$covariance$KroneckerProduct
MixedModelREML <- openreml$MixedModelREML

data <- agridat::john.alpha
n <- nrow(data)

y <- torch$tensor(data$yield, dtype = torch$float32)
X <- model.matrix(~ rep, data = data) |>
    torch$tensor(dtype = torch$float32)

Z_gen <- model.matrix(~ 0 + gen, data = data) |>
    torch$tensor(dtype = torch$float32)
Z_rep_block <- model.matrix(~ 0 + block:rep, data = data) |>
    torch$tensor(dtype = torch$float32)

reml <- MixedModelREML(
    genotype = list(Z_gen, ScalarMatrix(data$gen)),
    replicate_block = list(
        Z_rep_block,
        KroneckerProduct(
            G_rep = IdentityMatrix(data$rep),
            G_block = ScalarMatrix(data$block)
        )
    ),
    R = ScalarMatrix(n)
)
print(reml)

result <- reml$optimize(y, X, torch$zeros(3L), verbose = 1L)
print(py_to_r(reml$build_params()$numpy()))

print(py_to_r(reml$blue(y, X)$numpy()))

print(py_to_r(reml$blup(y, X)$numpy()))

gen_blup <- reml$blup_per_term(y, X)[["genotype"]]
print(setNames(py_to_r(gen_blup$numpy()), levels(data$gen)))
