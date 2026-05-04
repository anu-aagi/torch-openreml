#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch_openreml.covariance import DesignMatrix

mat = DesignMatrix(torch.tensor([1.0, 2.0, 3.0, 4.0]))
print(mat())

mat = DesignMatrix(["a", "b", "a", "c"])
print(mat())

mat = DesignMatrix(["a", "b", "a", "c"], levels=["c", "b", "a"])
print(mat())

