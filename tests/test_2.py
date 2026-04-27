import torch_openreml
import torch

A = torch_openreml.covariance.ScalarMatrix(3)
A.build(torch.tensor([1.0]))
manual_grad = A.grad
A.auto_grad(torch.tensor([1.0]))
auto_grad = A.grad

print(manual_grad)
print(torch.allclose(manual_grad, auto_grad))
print(manual_grad.shape == auto_grad.shape)

A = torch_openreml.covariance.DiagonalMatrix(3)
A.build(torch.tensor([1.0, 2.0, 3.0]))
manual_grad = A.grad
A.auto_grad(torch.tensor([1.0, 2.0, 3.0]))
auto_grad = A.grad

print(manual_grad)
print(torch.allclose(manual_grad, auto_grad))
print(manual_grad.shape == auto_grad.shape)

A = torch_openreml.covariance.CompoundSymmetricMatrix(3)
A.build(torch.tensor([1.0, 2.0]))
manual_grad = A.grad
A.auto_grad(torch.tensor([1.0, 2.0]))
auto_grad = A.grad

print(manual_grad)
print(torch.allclose(manual_grad, auto_grad))
print(manual_grad.shape == auto_grad.shape)
print(A.trans_params(torch.tensor([1.0, 2.0])))

A = torch_openreml.covariance.AR1Matrix(3)
A.build(torch.tensor([1.0, 2.0]))
manual_grad = A.grad
A.auto_grad(torch.tensor([1.0, 2.0]))
auto_grad = A.grad

print(manual_grad)
print(torch.allclose(manual_grad, auto_grad))
print(manual_grad.shape == auto_grad.shape)
print(A.trans_params(torch.tensor([1.0, 2.0])))

A.build({"sigma^2": torch.tensor([1.0]), "rho": torch.tensor([2.0])})

A = torch.eye(3, dtype=torch.float32, device=torch.device("cpu"))
B = torch_openreml.covariance.ScalarMatrix(3)
C = torch_openreml.covariance.DiagonalMatrix(3)
D = torch_openreml.covariance.Sum({"A": A, "B": B, "C": C})
D
D.param_names

D.build(torch.tensor([0.0, 1.0, 2.0, 3.0]))
manual_grad = D.grad
D.auto_grad(torch.tensor([0.0, 1.0, 2.0, 3.0]))
auto_grad = D.grad

print(manual_grad)
print(torch.allclose(manual_grad, auto_grad))
print(manual_grad.shape == auto_grad.shape)
print(D.grad_names)

D.build_operands(torch.tensor([0.0, 1.0, 2.0, 3.0]))

E = torch_openreml.covariance.Sum({"D.1": D, "D.2": D})
E
E.param_names

E.build(torch.tensor([0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0]))
manual_grad = E.grad
E.auto_grad(torch.tensor([0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0]))
auto_grad = E.grad

print(manual_grad)
print(torch.allclose(manual_grad, auto_grad))
print(manual_grad.shape == auto_grad.shape)

Z = torch.cat([torch.eye(3), torch.eye(3)])
G = torch_openreml.covariance.ScalarMatrix(3)
S = torch_openreml.covariance.LinearPropagation({"Z": Z, "G": G})
S
S.param_names
S.build(torch.tensor([1.0]))
manual_grad = S.grad
S.auto_grad(torch.tensor([1.0]))
auto_grad = S.grad

print(manual_grad)
print(torch.allclose(manual_grad, auto_grad))
print(manual_grad.shape == auto_grad.shape)
print(S.grad_names)

A = torch_openreml.covariance.DiagonalMatrix(3)
B = torch.eye(2)
C = torch_openreml.covariance.KroneckerProduct({"A": A, "B": B})
C.build(torch.tensor([0.0, 1.0, 2.0]))
C.grad
C.auto_grad(torch.tensor([0.0, 1.0, 2.0]))
C.grad
