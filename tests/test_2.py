import torch_openreml
import torch

A = torch_openreml.covariance.ScalarMatrix(3)
x = torch.tensor([1.0])
A(x)
manual_grad = A.manual_grad(x)[0]
auto_grad = A.auto_grad(x)[0]
print(torch.allclose(manual_grad, auto_grad))
print(manual_grad.shape == auto_grad.shape)

A = torch_openreml.covariance.DiagonalMatrix(3)
x = torch.tensor([1.0, 2.0, 3.0])
A(x)
manual_grad = A.manual_grad(x)[0]
auto_grad = A.auto_grad(x)[0]
print(torch.allclose(manual_grad, auto_grad))
print(manual_grad.shape == auto_grad.shape)

A = torch_openreml.covariance.CompoundSymmetricMatrix(3)
x = torch.tensor([1.0, 2.0])
A(x)
manual_grad = A.manual_grad(x)[0]
auto_grad = A.auto_grad(x)[0]
print(torch.allclose(manual_grad, auto_grad))
print(manual_grad.shape == auto_grad.shape)

A = torch_openreml.covariance.AR1Matrix(3)
x = torch.tensor([1.0, 2.0])
A(x)
manual_grad = A.manual_grad(x)[0]
auto_grad = A.auto_grad(x)[0]
print(torch.allclose(manual_grad, auto_grad))
print(manual_grad.shape == auto_grad.shape)

A({"sigma^2": torch.tensor([1.0]), "rho": torch.tensor([2.0])})


A = torch.eye(3, dtype=torch.float32, device=torch.device("cpu"))
B = torch_openreml.covariance.ScalarMatrix(3)
C = torch_openreml.covariance.DiagonalMatrix(3)
D = torch_openreml.covariance.Sum({"A": A, "B": B, "C": C})
D
D.param_names

x = torch.tensor([0.0, 1.0, 2.0, 3.0])
D(x)
manual_grad = D.manual_grad(x)[0]
auto_grad = D.auto_grad(x)[0]
print(manual_grad)
print(torch.allclose(manual_grad, auto_grad))
print(manual_grad.shape == auto_grad.shape)
print(D.manual_grad(x)[1])

D.build_operands(torch.tensor([0.0, 1.0, 2.0, 3.0]))
D.operands_grad(torch.tensor([0.0, 1.0, 2.0, 3.0]))

E = torch_openreml.covariance.Sum({"D.1": D, "D.2": D})
E
E.param_names

x = torch.tensor([0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0])
E(x)
manual_grad = E.manual_grad(x)[0]
auto_grad = E.auto_grad(x)[0]

print(manual_grad)
print(torch.allclose(manual_grad, auto_grad))
print(manual_grad.shape == auto_grad.shape)

Z = torch.cat([torch.eye(3), torch.eye(3)])
G = torch_openreml.covariance.ScalarMatrix(3)
S = torch_openreml.covariance.LinearPropagation({"Z": Z, "G": G})
S
S.param_names
x = torch.tensor([1.0])
S(x)
manual_grad = S.manual_grad(x)[0]
auto_grad = S.auto_grad(x)[0]

print(manual_grad)
print(torch.allclose(manual_grad, auto_grad))
print(manual_grad.shape == auto_grad.shape)
print(S.manual_grad(x)[1])

A = torch_openreml.covariance.DiagonalMatrix(4)
B = torch.eye(2)
C = torch_openreml.covariance.KroneckerProduct({"A": A, "B": B})
C(torch.tensor([0.0, 1.0, 2.0, 3.0]))
C.manual_grad(torch.tensor([0.0, 1.0, 2.0, 3.0]))
C.auto_grad(torch.tensor([0.0, 1.0, 2.0, 3.0]))
