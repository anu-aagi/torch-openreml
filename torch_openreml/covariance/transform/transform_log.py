from torch_openreml.covariance.parameter.transform import Transform
import torch

class TransformLog(Transform):
    def __init__(self):
        pass
    
    def __call__(self, x):
        return torch.log(x)
    
    def inverse(self, x):
        return torch.exp(x)
    
    def chain_rule_factor(self, x):
        return torch.exp(x)

class TransformLog2(Transform):
    def __init__(self):
        pass

    def __call__(self, x):
        return torch.log2(x)

    def inverse(self, x):
        return torch.exp2(x)

    def chain_rule_factor(self, x):
        return torch.exp2(x) * torch.log(torch.tensor([2], dtype=x.dtype, device=x.device))

class TransformLog10(Transform):
    def __init__(self):
        pass

    def __call__(self, x):
        return torch.log10(x)

    def inverse(self, x):
        return torch.pow(10.0, x)
    torch.exp

    def chain_rule_factor(self, x):
        return torch.pow(10.0, x) * torch.log(torch.tensor([10], dtype=x.dtype, device=x.device))
