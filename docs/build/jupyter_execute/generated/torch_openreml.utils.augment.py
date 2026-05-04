#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch_openreml.utils import augment

x1 = torch.ones(4, 2)
x2 = torch.zeros(4, 3)
augment(x1, x2)

