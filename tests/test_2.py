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
print(manual_grad.shape == auto_grad.shape)

A = torch_openreml.covariance.DiagonalMatrix(3)
A.build(torch.tensor([1.0, 2.0, 3.0]))
manual_grad = A.grad
A.auto_grad(torch.tensor([1.0, 2.0, 3.0]))
auto_grad = A.grad

print(manual_grad)
print(manual_grad == auto_grad)
print(manual_grad.shape == auto_grad.shape)

A = torch_openreml.covariance.CompoundSymmetricMatrix(3)
A.build(torch.tensor([1.0, 2.0]))
manual_grad = A.grad
A.auto_grad(torch.tensor([1.0, 2.0]))
auto_grad = A.grad

print(manual_grad)
print(manual_grad == auto_grad)
print(manual_grad.shape == auto_grad.shape)
print(A.trans_rho(torch.tensor([2.0])))

A = torch_openreml.covariance.AR1Matrix(3)
A.build(torch.tensor([1.0, 2.0]))
manual_grad = A.grad
A.auto_grad(torch.tensor([1.0, 2.0]))
auto_grad = A.grad

print(manual_grad)
print(manual_grad == auto_grad)
print(manual_grad.shape == auto_grad.shape)
print(A.trans_rho(torch.tensor([2.0])))

A.build({"log_sigma": torch.tensor([1.0]), "scaled_rho": torch.tensor([2.0])})

A = torch.eye(3)
B = torch_openreml.covariance.ScalarMatrix(3)
C = torch_openreml.covariance.DiagonalMatrix(3)
D = torch_openreml.covariance.Sum({"A": A, "B": B, "C": C})
D.param_names

D.build(torch.tensor([0.0, 1.0, 2.0, 3.0]))
manual_grad = D.grad
D.auto_grad(torch.tensor([0.0, 1.0, 2.0, 3.0]))
auto_grad = D.grad

print(manual_grad)
print(manual_grad == auto_grad)
print(manual_grad.shape == auto_grad.shape)

D.build_operands(torch.tensor([0.0, 1.0, 2.0, 3.0]))
