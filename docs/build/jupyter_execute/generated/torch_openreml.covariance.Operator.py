#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch_openreml.covariance import Sum, ScalarMatrix

x = Sum(ScalarMatrix(2), ScalarMatrix(2))
print(x)

x = Sum(A = ScalarMatrix(2), B = ScalarMatrix(2))
print(x)

print(x.param_names)

print(x(torch.zeros(2)))
print(x({"A/sigma^2": torch.zeros(1), "B/sigma^2": torch.zeros(1)}))


# In[2]:


import torch
from torch_openreml.covariance import Sum, ScalarMatrix

x = Sum(ScalarMatrix(2), ScalarMatrix(2))
print(x.trans_params(torch.zeros(2)))


# In[3]:


import torch
from torch_openreml.covariance import Sum, ScalarMatrix

x = Sum(ScalarMatrix(2), ScalarMatrix(2))
v_groups = x.build_operands(torch.tensor([1.0, 2.0]))
print(v_groups[0])
print(v_groups[1])


# In[4]:


import torch
from torch_openreml.covariance import Sum, ScalarMatrix

x = Sum(ScalarMatrix(2), ScalarMatrix(2))
grad_groups, grad_name_groups = x.operands_grad(torch.tensor([1.0, 2.0]))
print(grad_groups[0])
print(grad_groups[1])
print(grad_name_groups[0])
print(grad_name_groups[1])

