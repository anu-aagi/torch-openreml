#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch_openreml.covariance.transform import TransformSigmoid

t = TransformSigmoid()
x = torch.tensor([-2.0, 0.0, 2.0])
t(x)


# In[2]:


import torch
from torch_openreml.covariance.transform import TransformSigmoid

t = TransformSigmoid()
x = torch.tensor([0.1, 0.5, 0.9])
t.inverse(x)


# In[3]:


import torch
from torch_openreml.covariance.transform import TransformSigmoid

t = TransformSigmoid()
x = torch.tensor([0.0, 1.0])
t.grad(x)

