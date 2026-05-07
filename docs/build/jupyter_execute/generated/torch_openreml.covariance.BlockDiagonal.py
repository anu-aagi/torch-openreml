#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch_openreml.covariance import ScalarMatrix, DiagonalMatrix, BlockDiagonal

block = BlockDiagonal(
    residual=ScalarMatrix(3),
    random=DiagonalMatrix(2)
)
params = torch.tensor([0.5, 0.0, 1.0])
block(params)

