#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch_openreml
print(torch_openreml.example_data.john_alpha)


# In[2]:


import torch
from torch_openreml import REML
from torch_openreml.utils import augment, n_distinct
from torch_openreml.covariance import DummyMatrix, IdentityMatrix, ScalarMatrix, Sum, CovariancePropagation, KroneckerProduct
from torch_openreml.example_data import john_alpha


# In[3]:


y = torch.tensor(john_alpha["yield"].values)
X = augment(torch.ones(len(john_alpha), 1),
            DummyMatrix(john_alpha["rep"], drop_first=True)())

Z_gen = DummyMatrix(john_alpha["gen"])
Z_rep_block = DummyMatrix(john_alpha["rep"], john_alpha["block"])

G_gen = ScalarMatrix(n_distinct(john_alpha["gen"]))
G_rep = IdentityMatrix(n_distinct(john_alpha["rep"]))
G_block = ScalarMatrix(n_distinct(john_alpha["block"]))

R = ScalarMatrix(len(john_alpha))

V = Sum(
    CovariancePropagation(Z_gen, G_gen),
    CovariancePropagation(
        Z_rep_block,
        KroneckerProduct(G_rep, G_block)
    ),
    R
)

print(V)


# In[4]:


reml = REML(V)
theta_hat, beta_hat, n_iter = reml.optimize(y, X, torch.zeros(3), verbose=2)
print(theta_hat, V.trans_params(theta_hat))
print(V.param_names)
print(beta_hat)

