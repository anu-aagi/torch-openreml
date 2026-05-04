#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch_openreml.utils import get_dtype

x = torch.tensor([1.0, 2.0])
y = torch.tensor([3.0, 4.0])
get_dtype(x, y)

