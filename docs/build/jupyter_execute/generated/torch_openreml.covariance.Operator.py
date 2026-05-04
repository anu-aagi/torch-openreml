#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch_openreml.covariance import DiagonalMatrix

mat = DiagonalMatrix(3)
mat.set_no_grad(index=0)
print(mat.no_grad_index)
print(mat.grad(torch.zeros(3)))


# In[2]:


import torch
from torch_openreml.covariance import DiagonalMatrix

mat = DiagonalMatrix(3)
params = torch.tensor([0.0, 0.5, 1.0])
mat.trans_params(params)

