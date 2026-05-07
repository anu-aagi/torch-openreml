#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch_openreml.covariance import EquicorrelationMatrix, HadamardProduct

n = 4
op = HadamardProduct(a=EquicorrelationMatrix(n), b=torch.tensor([5.0]))
params = torch.tensor([1.0])
op(params)

