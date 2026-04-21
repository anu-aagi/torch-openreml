from torch_openreml.covariance.transform.base import Transform
import torch

class TransformLogit(Transform):

    def __init__(self):
        pass

    def __call__(self, x):
        return torch.logit(x)

    def inverse(self, x):
        return torch.sigmoid(x)

    def chain_rule_factor(self, x):
        sigmoid = torch.sigmoid(x)
        return sigmoid * (1 - sigmoid)