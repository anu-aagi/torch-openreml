fit_lme4 <- lme4::lmer(yield ~ rep + (1|gen) + (1|rep:block),
                       data = agridat::john.alpha)

fit_lme4@beta
fit_lme4@theta
sigma(fit_lme4)
lme4::VarCorr(fit_lme4) |> 
  unlist() |> 
  unname()
