# torch-openreml

**torch-openreml** is an experimental Python library for fitting linear mixed-effects models using Average Information REML (AI-REML) with Torch.

## Overview

This package focuses purely on the computational backend of mixed model estimation. It implements the AI-REML algorithm using Torch for tensor operations and automatic differentiation of covariance structures, while parameter updates are carried out explicitly using the Average Information matrix.

Unlike traditional mixed-model software, **torch-openreml does not provide a formula interface, model parser, or user-friendly front end**. It is designed for users who want full control over model specification and are comfortable constructing model matrices and covariance structures programmatically.

## Note

The library may be **highly inefficient** due to its experimental nature and ongoing development.

## Philosophy

This library separates **computation** from **interface**:

- It does **not** implement formula syntax (e.g., `y ~ x + (1|g)`).
- It does **not** handle data preprocessing, design matrix construction, or model specification parsing.

## Model Formulation

The library assumes the standard linear mixed-effects model:


$$y = X\beta + Zb + \varepsilon$$

where:

- $y \in \mathbb{R}^n$: response vector  
- $X \in \mathbb{R}^{n \times p}$: fixed-effects design matrix  
- $\beta \in \mathbb{R}^p$: fixed-effects coefficients  
- $Z \in \mathbb{R}^{n \times q}$: random-effects design matrix  
- $b \in \mathbb{R}^q$: random effects  
- $\varepsilon \in \mathbb{R}^n$: residual errors  

### Distributional Assumptions


$$b \sim \mathcal{N}(0, G(\theta)), \quad \varepsilon \sim \mathcal{N}(0, R(\theta))$$

where both covariance structures are parameterized by $\theta$.

The marginal covariance of $y$ is:


$$\Sigma = Z G(\theta) Z^\top + R(\theta).$$


## Status

⚠️ **Experimental**

This library is under active development and should be considered experimental. Interfaces and implementations may change without backward compatibility guarantees.
