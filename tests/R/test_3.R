library(reticulate)

use_condaenv("torch-openreml")

fit_lme4 <- lme4::lmer(yield ~ rep + (1|gen) + (1|rep:block),
                       data = agridat::john.alpha)

lme4_beta <- fit_lme4@beta

lme4_sigma <- lme4::VarCorr(fit_lme4) |> 
    lapply(\(x) attr(x, "stddev")) |>
    unlist() |> 
    unname()
lme4_sigma <- c(lme4_sigma, sigma(fit_lme4))

openreml <- import("torch_openreml", convert = FALSE)
torch <- import("torch", convert = FALSE)

torch$set_default_device("cpu")
torch$set_default_dtype(torch$float32)


n <- nrow(agridat::john.alpha)
n_gen <- length(unique(agridat::john.alpha$gen))
n_rep <- length(unique(agridat::john.alpha$rep))
n_block <- length(unique(agridat::john.alpha$block))


IdentityMatrix <- openreml$covariance$IdentityMatrix
ScalarMatrix <- openreml$covariance$ScalarMatrix
LinearPropagation <- openreml$covariance$LinearPropagation
HadamardProduct <- openreml$covariance$HadamardProduct
Sum <- openreml$covariance$Sum

y <- torch$tensor(agridat::john.alpha$yield, dtype=torch$float32)

x <- torch$tensor(r_to_py(model.matrix(fit_lme4)), dtype = torch$float32)

G_gen <- GSidenCovariance(agridat::john.alpha$gen, ScalarMatrix)
G_rep <- GSidenCovariance(agridat::john.alpha$rep, IdentityMatrix)
G_block <- GSidenCovariance(agridat::john.alpha$block, ScalarMatrix)

G_rep_block <- HadamardProduct(list(G_rep = G_rep, G_block = G_block))

R <- ScalarMatrix(n)

V <- Sum(list(G_gen = G_gen, G_rep_block = G_rep_block, R = R))

print(V)
print(V$param_names)

fit_openreml <- openreml$REML(v_builder = V)

result <- fit_openreml$optimize(y, 
                                x, 
                                torch$zeros(3L), 
                                require_loglik = TRUE, 
                                verbose = 2,
                                eta = 1)

openreml_beta <- py_to_r(fit_openreml$get_beta()$numpy())
openreml_sigma <- exp(py_to_r(fit_openreml$get_theta()$numpy()))

print(lme4_beta - openreml_beta)
print(lme4_sigma - openreml_sigma)
