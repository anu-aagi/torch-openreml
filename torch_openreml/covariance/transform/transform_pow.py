from torch_openreml.covariance.transform.transform import Transform
import torch

class TransformPow(Transform):

    codomain = "\u211D"
    domain = "\u211D"

    def __init__(self, factor=2.0):
        self.factor = factor

    def __call__(self, x):
        return torch.pow(x, self.factor)

    def inverse(self, x):
        return torch.sqrt(x)

    def chain_rule_factor(self, x):
        return self.factor * x

    def __repr__(self):
        return f"{self.__class__.__name__}(factor={self.factor})"