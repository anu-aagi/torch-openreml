from torch_openreml.covariance.transform.transform import Transform
import torch


class TransformSqrtLog(Transform):
    domain = "\u211D\u2080\u207A"
    codomain = "\u211D"

    def __init__(self):
        pass

    def __call__(self, x):
        return torch.log(x) / 2.0

    def inverse(self, x):
        return torch.exp(2.0 * x)

    def chain_rule_factor(self, x):
        return 2 * torch.exp(2 * x)

