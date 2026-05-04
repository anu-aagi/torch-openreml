#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch_openreml.covariance import DiagonalMatrix

mat = DiagonalMatrix(3)
params = torch.tensor([0.0, 0.5, 1.0])
sigma2 = mat.trans_params(params)
mat.set_intermediates(params, {"sigma2": sigma2})
mat.get_intermediates(params)


# In[2]:


import torch
from torch_openreml.covariance import DiagonalMatrix

mat = DiagonalMatrix(3)
params = torch.tensor([0.0, 0.5, 1.0])
sigma2 = mat.trans_params(params)
mat.set_intermediates(params, {"sigma2": sigma2})
mat.get_intermediates(params)


# In[3]:


import torch
from torch_openreml.covariance import DiagonalMatrix

mat = DiagonalMatrix(3)
params = torch.tensor([0.0, 0.5, 1.0])
sigma2 = mat.trans_params(params)
mat.set_intermediates(params, {"sigma2": sigma2})
print(mat.get_intermediates(params))
mat.reset_intermediates()
print(mat.get_intermediates(params))


# In[4]:


import torch
from torch_openreml.covariance import DiagonalMatrix

mat = DiagonalMatrix(3)
mat.set_no_grad(index=0)
print(mat.no_grad_index)
print(mat.grad(torch.zeros(3)))


# In[5]:


import torch
from torch_openreml.covariance import DiagonalMatrix

mat = DiagonalMatrix(3)
param_dict = {"sigma^2_0": torch.tensor([0.0]),
              "sigma^2_1": torch.tensor([0.5]),
              "sigma^2_2": torch.tensor([1.0])}
mat.from_param_dict(param_dict)


# In[6]:


import torch
from torch_openreml.covariance import DiagonalMatrix

mat = DiagonalMatrix(3)
params = torch.tensor([0.0, 0.5, 1.0])
mat.to_param_dict(params)


# In[7]:


import torch
from torch_openreml.covariance import DiagonalMatrix

mat = DiagonalMatrix(3)
params = torch.tensor([0.0, 0.5, 1.0])
mat.trans_params(params)


# In[8]:


import torch
from torch_openreml.covariance import DiagonalMatrix

mat = DiagonalMatrix(3)
params = torch.tensor([0.0, 0.5, 1.0])
mat.trans_grad(params)


# In[9]:


import torch
from torch_openreml.covariance import DiagonalMatrix

mat = DiagonalMatrix(2)
params = torch.tensor([0.0, 0.5])
grad, grad_names = mat.auto_grad(params)
grad, grad_names


# In[10]:


import torch
from torch_openreml.covariance import DiagonalMatrix

mat = DiagonalMatrix(2)
params = torch.tensor([0.0, 0.5])
grad, grad_names = mat.grad(params)
grad, grad_names

