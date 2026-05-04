#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch_openreml.covariance.transform import TransformExp, TransformPow, TransformChain

t = TransformChain([TransformExp(), TransformPow(factor=2.0)])
x = torch.tensor([1.0])
t(x)


# In[2]:


import torch
from torch_openreml.covariance.transform import TransformExp, TransformPow, TransformChain

t = TransformChain([TransformExp(), TransformPow(factor=2.0)])
x = torch.tensor([4.0])
t.inverse(x)


# In[3]:


import torch
from torch_openreml.covariance.transform import TransformExp, TransformPow, TransformChain

t = TransformChain([TransformExp(), TransformPow(factor=2.0)])
x = torch.tensor([1.0])
t.grad(x)

