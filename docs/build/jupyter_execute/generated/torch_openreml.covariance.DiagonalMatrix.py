#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch_openreml.covariance import DiagonalMatrix

mat = DiagonalMatrix(3)
params = torch.tensor([0.0, 0.5, 1.0])
print(mat(params))

print(mat.grad(params))

