#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch_openreml import REML
from torch_openreml.covariance import ScalarMatrix

n, p = 50, 2
y = torch.randn(n)
x = torch.randn(n, p)
theta = torch.tensor([0.0])

mat = ScalarMatrix(n)
reml = REML(v_builder=mat)
theta_hat, beta_hat, n_iter = reml.optimize(y, x, theta, verbose=2)
theta_hat, beta_hat

