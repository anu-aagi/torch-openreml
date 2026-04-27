from torch_openreml.covariance.transform.transform import Transform

class TransformScaleShift(Transform):

    domain = "\u211D"
    codomain = "\u211D"

    def __init__(self, a, b=0.0):
        self.a = a
        self.b = b

    def __call__(self, x):
        return self.a * x + self.b

    def inverse(self, x):
        return (x - self.b) / self.a

    def chain_rule_factor(self, x):
        return self.a

    def __repr__(self):
        return f"{self.__class__.__name__}(a={self.a}, b={self.b})"
