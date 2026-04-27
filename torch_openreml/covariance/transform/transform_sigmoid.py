from torch_openreml.covariance.transform.transform import Transform
import torch

class TransformSigmoid(Transform):

    domain = "\u211D\u2080\u207A"
    codomain = "(0, 1)"

    def __init__(self):
        pass

    def __call__(self, x):
        return torch.sigmoid(x)

    def inverse(self, x):
        return torch.logit(x)

    def chain_rule_factor(self, x):
        sigmoid = torch.sigmoid(x)
        return sigmoid * (1 - sigmoid)