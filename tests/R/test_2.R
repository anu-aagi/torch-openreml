library(reticulate)

use_condaenv("torch-openreml")

fit_lme4 <- lme4::lmer(yield ~ rep + (1|gen),
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

y <- torch$tensor(agridat::john.alpha$yield, dtype=torch$float32)
x <- torch$tensor(r_to_py(model.matrix(fit_lme4)), dtype = torch$float32)
z_gen <- torch$tensor(r_to_py(model.matrix(~ 0 + gen, data = agridat::john.alpha)), dtype = torch$float32)

g_gen <- openreml$covariance$ScalarMatrix(z_gen$shape[1])
g <- openreml$covariance$LinearPropagation(list(z_gen = z_gen, g_gen = g_gen))
r <- openreml$covariance$ScalarMatrix(z_gen$shape[0])

v <- openreml$covariance$Sum(list(g = g, r = r))
print(v)
print(v$param_names)

fit_openreml <- openreml$REML(v_model = v)

result <- fit_openreml$optimize(y, 
                                x, 
                                torch$zeros(2L), 
                                require_loglik = TRUE, 
                                verbose = 2,
                                eta = 1)

openreml_beta <- py_to_r(fit_openreml$get_beta()$numpy())
openreml_sigma <- exp(py_to_r(fit_openreml$get_theta()$numpy()))

print(lme4_beta - openreml_beta)
print(lme4_sigma - openreml_sigma)
