#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch_openreml.covariance import DiagonalMatrix, CovariancePropagation

n, q = 6, 3
z = torch.randn(n, q)
g = DiagonalMatrix(q)
op = CovariancePropagation(z=z, g=g)
params = torch.tensor([0.0, 0.5, 1.0])
op(params)

