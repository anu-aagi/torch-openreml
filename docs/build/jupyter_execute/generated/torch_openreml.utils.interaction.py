#!/usr/bin/env python
# coding: utf-8

# In[1]:


from torch_openreml.utils import interaction

a = ["control", "control", "treatment"]
b = ["male", "female", "male"]
interaction(a, b)

