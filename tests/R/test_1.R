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

y <- torch$tensor(agridat::john.alpha$yield, dtype=torch$float32)
x <- torch$tensor(r_to_py(model.matrix(fit_lme4)), dtype = torch$float32)
z_gen <- torch$tensor(r_to_py(model.matrix(~ 0 + gen, data = agridat::john.alpha)), dtype = torch$float32)
k_gen <- z_gen %*% z_gen$T
z_rb <- torch$tensor(r_to_py(model.matrix(~ 0 + rep:block, data = agridat::john.alpha)), dtype = torch$float32)
k_rb <- z_rb %*% z_rb$T
i_n <- torch$eye(nrow(agridat::john.alpha), dtype = torch$float32)

map_theta_to_v <- function(theta) {
  
  sigma <- torch$exp(theta)
  sigma2 <- sigma * sigma
  
  v <- k_gen * sigma2[0] + k_rb * sigma2[1] + sigma2[2] * i_n
  
  return(v)
}

fit_openreml <- openreml$REML(map_theta_to_v = map_theta_to_v)
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

