from torch_openreml.covariance.transform.base import Transform
import torch

class TransformSqrt(Transform):

    domain = "\u211D\u2080\u207A"
    codomain = "\u211D\u2080\u207A"

    def __init__(self):
        pass

    def __call__(self, x):
        return torch.sqrt(x)

    def inverse(self, x):
        return torch.pow(x, 2.0)

    def chain_rule_factor(self, x):
        return 2 * x