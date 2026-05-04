#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch_openreml.utils import numeric_to_design_matrix

x1 = torch.tensor([1.0, 2.0, 3.0])
x2 = torch.tensor([4.0, 5.0, 6.0])
numeric_to_design_matrix(x1, x2)

