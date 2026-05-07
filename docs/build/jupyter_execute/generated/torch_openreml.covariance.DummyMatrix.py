#!/usr/bin/env python
# coding: utf-8

# In[1]:


from torch_openreml.covariance import DummyMatrix

rep = ["rep1", "rep2", "rep2"]
block = ["block1", "block2", "block1"]

mat = DummyMatrix(rep, block)
print(mat())
print(mat.colnames)

mat = DummyMatrix(rep, block, drop_first=True)
print(mat())
print(mat.colnames)

