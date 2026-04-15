import torch_openreml
import torch

A = torch_openreml.covariance.IdentityMatrix(3)
A.build(torch.tensor([1.0]))

A = torch_openreml.covariance.ScalarMatrix(3)
A.build(torch.tensor([1.0]))
manual_grad = A.grad
A.auto_grad(torch.tensor([1.0]))
auto_grad = A.grad

print(manual_grad)
print(manual_grad == auto_grad)

A = torch_openreml.covariance.DiagonalMatrix(3)
A.build(torch.tensor([1.0, 2.0, 3.0]))
manual_grad = A.grad
A.auto_grad(torch.tensor([1.0, 2.0, 3.0]))
auto_grad = A.grad

print(manual_grad)
print(manual_grad == auto_grad)

A = torch_openreml.covariance.CompoundSymmetricMatrix(3)
A.build(torch.tensor([1.0, 2.0]))
manual_grad = A.grad
A.auto_grad(torch.tensor([1.0, 2.0]))
auto_grad = A.grad

print(manual_grad)
print(manual_grad == auto_grad)
print(A.trans_rho(torch.tensor([2.0])))

A = torch_openreml.covariance.AR1Matrix(3)
A.build(torch.tensor([1.0, 2.0]))
manual_grad = A.grad
A.auto_grad(torch.tensor([1.0, 2.0]))
auto_grad = A.grad

print(manual_grad)
print(manual_grad == auto_grad)
print(A.trans_rho(torch.tensor([2.0])))

