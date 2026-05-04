#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch_openreml.utils import categorical_to_design_matrix

print(categorical_to_design_matrix(["a", "b", "a", "c"]))

print(categorical_to_design_matrix(["a", "b", "a", "c"], drop_first=True))

