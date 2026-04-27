from torch_openreml.covariance.transform.transform import Transform
import torch

class TransformExp(Transform):

    domain = "\u211D\u207A"
    codomain = "\u211D"

    def __init__(self):
        pass
    
    def __call__(self, x):
        return torch.log(x)
    
    def inverse(self, x):
        return torch.exp(x)
    
    def chain_rule_factor(self, x):
        return torch.exp(x)

class TransformLog2(Transform):

    domain = "\u211D\u207A"
    codomain = "\u211D"

    def __init__(self):
        pass

    def __call__(self, x):
        return torch.log2(x)

    def inverse(self, x):
        return torch.exp2(x)

    def chain_rule_factor(self, x):
        return torch.exp2(x) * torch.log(torch.tensor([2], dtype=x.dtype, device=x.device))

class TransformLog10(Transform):

    domain = "\u211D\u207A"
    codomain = "\u211D"

    def __init__(self):
        pass

    def __call__(self, x):
        return torch.log10(x)

    def inverse(self, x):
        return torch.pow(10.0, x)

    def chain_rule_factor(self, x):
        return torch.pow(10.0, x) * torch.log(torch.tensor([10], dtype=x.dtype, device=x.device))
