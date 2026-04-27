from abc import ABC, abstractmethod

class Transform(ABC):

    domain = "\u211D"
    codomain = "\u211D"
    
    @abstractmethod
    def __call__(self, x):
        pass
    
    @abstractmethod
    def inverse(self, x):
        pass
    
    @abstractmethod
    def chain_rule_factor(self, x):
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}()"

    def __str__(self):
        return f"{self.__class__.__name__}: {self.domain} \u21A6 {self.codomain}"

class TransformChain(Transform):
    def __init__(self, chain):
        if not isinstance(chain, (list, tuple)):
            chain = [chain]
        self.chain = chain

        for trans in chain:
            if not isinstance(trans, Transform):
                raise TypeError("Chain needs to be a list or a tuple of Transform objects!")

    def __call__(self, x):
        for trans in self.chain:
            x = trans(x)
        return x

    def inverse(self, x):
        for trans in reversed(self.chain):
            x = trans.inverse(x)

    def chain_rule_factor(self, x):
        factor = None
        for trans in self.chain:
            if factor is None:
                factor = trans.chain_rule_factor(x)
            else:
                factor = factor * trans.chain_rule_factor(x)

            x = trans(x)

        return factor

    def __repr__(self):
        return f"{self.__class__.__name__}({self.chain!r})"

    def __str__(self):
        return f"{self.__class__.__name__}({self.chain!r})"
