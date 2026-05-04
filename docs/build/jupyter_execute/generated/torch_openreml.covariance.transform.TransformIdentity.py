#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch_openreml.covariance.transform import TransformIdentity

t = TransformIdentity()
x = torch.tensor([0.0, 1.0, -3.5])
t(x)


# In[2]:


import torch
from torch_openreml.covariance.transform import TransformIdentity

t = TransformIdentity()
x = torch.tensor([2.0, -1.0])
t.inverse(x)


# In[3]:


import torch
from torch_openreml.covariance.transform import TransformIdentity

t = TransformIdentity()
x = torch.tensor([0.0])
t.grad(x)

