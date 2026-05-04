#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch_openreml.covariance.transform import TransformPow

t = TransformPow(factor=3.0)
x = torch.tensor([1.0, 2.0, 3.0])
t(x)


# In[2]:


import torch
from torch_openreml.covariance.transform import TransformPow

t = TransformPow(factor=2.0)
x = torch.tensor([1.0, 4.0, 9.0])
t.inverse(x)


# In[3]:


import torch
from torch_openreml.covariance.transform import TransformPow

t = TransformPow(factor=3.0)
x = torch.tensor([2.0, 3.0])
t.grad(x)

