#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch_openreml.covariance import AR1Matrix, ScalarMatrix, KroneckerProduct

op = KroneckerProduct(time=AR1Matrix(2), subject=ScalarMatrix(2))
params = torch.tensor([1.0, 1.0, 1.0])
op(params)

