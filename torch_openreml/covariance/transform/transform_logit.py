from torch_openreml.covariance.transform.transform import Transform
import torch

class TransformLogit(Transform):

    domain = "(0, 1)"
    codomain = "\u211D\u2080\u207A"

    def __init__(self):
        pass

    def __call__(self, x):
        return torch.logit(x)

    def inverse(self, x):
        return torch.sigmoid(x)

    def chain_rule_factor(self, x):
        sigmoid = torch.sigmoid(x)
        return sigmoid * (1 - sigmoid)