<p align="center">
  <a href="https://github.com/anu-aagi/torch-openreml/blob/main/docs/source/_static/hex-icon-readme.png">
    <img src="https://raw.githubusercontent.com/anu-aagi/torch-openreml/main/docs/source/_static/hex-icon-readme.png" alt="torch-openreml" width="200">
  </a>
</p>

# torch-openreml

<p>
  <img src="https://img.shields.io/badge/dev-0.4.0--alpha-blue" alt="Development version 0.4.0-alpha">
  <img src="https://img.shields.io/pypi/v/torch-openreml?include_prereleases" alt="PyPI version">
  <img src="https://img.shields.io/badge/license-GPLv3-blue" alt="GPL-3.0">
  <img src="https://img.shields.io/badge/python-%3E%3D3.10-blue" alt="Python >=3.10">
  <img src="https://img.shields.io/badge/pytorch-%3E%3D2.0-orange" alt="PyTorch >=2.0">
  <img src="https://img.shields.io/badge/status-experimental-yellow" alt="Experimental">
</p>

**torch-openreml** is a PyTorch-based library for AI-REML estimation of linear mixed models.

**Author & Maintainer:** Weihao (Patrick) Li — patrick.li@anu.edu.au

## Overview

torch-openreml fits linear mixed-effects models using the Average Information REML (AI-REML)
algorithm on a PyTorch backend. Covariance structures are specified through a modular system of
matrices and operators, with automatic or manually specified gradients and optional parameter
transformations for constrained estimation.

Unlike traditional mixed-model software, it does not provide a formula interface. Fixed- and
random-effects design matrices and covariance structures are defined directly in code, so the
library stays focused on the computational and optimisation backend rather than model
specification syntax.

## Features

- **Torch-based backend** — runs on CPU, GPU, and other available accelerators.
- **AI-REML estimation engine** — variance components by quasi-Newton optimisation on the Average Information matrix.
- **Extensible covariance structure** — composable matrices and operators, built-in or user-defined.
- **Hybrid differentiation** — automatic differentiation or manually specified gradients.
- **Composable parameter transformations** — chainable pipelines for flexible parameterisation.

## Installation

Install the latest development version directly from GitHub:

```bash
pip install git+https://github.com/anu-aagi/torch-openreml.git
```

**Dependencies:** `torch>=2.0`, `pandas`, `tqdm` — Python 3.10 or later.

## Model

The library assumes the standard linear mixed-effects model

$$y = X\beta + Zb + \varepsilon$$

with

$$b \sim \mathcal{N}(0, G(\theta)), \qquad \varepsilon \sim \mathcal{N}(0, R(\theta)),$$

so that the marginal covariance of $y$ is

$$V = Z G(\theta) Z^\top + R(\theta).$$

## Quick start

The example below fits `yield` from the bundled `john_alpha` dataset — a resolvable alpha lattice
trial — with a fixed effect for `rep`, a random intercept for `gen`, and a random `rep:block`
interaction.

```python
import torch
from torch_openreml import MarginalREML, blup
from torch_openreml.covariance import (
    Augment, BlockDiagonal, DummyMatrix, IdentityMatrix,
    KroneckerProduct, ScalarMatrix, Sum, CovariancePropagation,
)
from torch_openreml.example_data import john_alpha

y = torch.tensor(john_alpha["yield"].values)

# Fixed effects X = [1 | rep], with the first replicate absorbed by the intercept.
X = Augment(torch.ones(len(john_alpha), 1),
            DummyMatrix(john_alpha["rep"], drop_first=True))()

# Marginal covariance V = Z G Z^T + R, where Z = [gen | rep:block] is the
# random-effect design, G = diag(gen, rep:block) has independent gen effects and
# independent blocks within each replicate, and R = sigma2_e I is the residual.
V = Sum(
    random=CovariancePropagation(
        Z=Augment(
            Z_gen=DummyMatrix(john_alpha["gen"]),
            Z_rep_block=DummyMatrix(john_alpha["rep"], john_alpha["block"]),
        ),
        G=BlockDiagonal(
            G_gen=ScalarMatrix(john_alpha["gen"]),
            G_rep_block=KroneckerProduct(
                IdentityMatrix(john_alpha["rep"]),
                ScalarMatrix(john_alpha["block"]),
            ),
        ),
    ),
    residual=ScalarMatrix(len(john_alpha)),
)

# Fit from zero starting values for the three variance components (gen, rep:block,
# residual), then print the fitted variances and the fixed effects.
reml = MarginalREML(V)
theta_hat, beta_hat, n_iter = reml.optimize(y, X, torch.zeros(3), verbose=1)
print(V.build_params(theta_hat), beta_hat)

# BLUP of the random effects, jointly across all levels or for one component.
tree, _ = V.tree(theta_hat)
b_hat = blup(y, X, tree["random/Z"], tree["random/G"], tree["/"])
gen_blup = blup(y, X, tree["random/Z/Z_gen"], tree["random/G/G_gen"], tree["/"])
```

Fitting happens on the transformed parameter scale; `V.build_params` maps the estimates back to
variance components, named by `V.free_param_names`. `V.tree(theta_hat)` evaluates every node of the
covariance structure and keys the results by operand path, which is how the BLUP reaches the design
and covariance matrices: `b_hat` holds one prediction per genotype followed by one per
replicate-by-block combination, `gen_blup` the genotype predictions alone.

## Documentation

Full documentation: **https://torch-openreml.patrickli.org**

- [Vignettes](https://torch-openreml.patrickli.org/vig) — worked examples, starting with the Marginal REML workflow.
- [Technical Documentation](https://torch-openreml.patrickli.org/tech) — model formulation, REML and ML theory, score and AI matrix derivations.
- [API Reference](https://torch-openreml.patrickli.org/api) — `MarginalREML`, covariance matrices, operators, transforms, utilities, and example datasets.
- [For R Users](https://torch-openreml.patrickli.org/r_user)
- [Changelog](https://torch-openreml.patrickli.org/change_log)

## Citing

```bibtex
@software{torch_openreml,
  author = {Weihao Li},
  title  = {torch-openreml},
  year   = {2026},
  url    = {https://github.com/anu-aagi/torch-openreml/}
}
```

## Status

⚠️ **Experimental.** The library is under active development. Interfaces and implementations may
change without backward-compatibility guarantees, and it may be slightly inefficient.
