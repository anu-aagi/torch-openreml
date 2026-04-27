from torch_openreml.covariance.transform.transform import Transform

class TransformIdentity(Transform):
    domain = "\u211D"
    codomain = "\u211D"

    def __init__(self):
        pass

    def __call__(self, x):
        return x

    def inverse(self, x):
        return x

    def chain_rule_factor(self, x):
        return 1.0

