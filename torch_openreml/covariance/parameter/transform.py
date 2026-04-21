from abc import ABC, abstractmethod

class Transform(ABC):
    
    @abstractmethod
    def __call__(self, x):
        pass
    
    @abstractmethod
    def inverse(self, x):
        pass
    
    @abstractmethod
    def chain_rule_factor(self, x):
        pass

class TransformChain:
    def __init__(self, chain):
        if not isinstance(chain, list):
            chain = [chain]
        self.chain = chain

        for trans in chain:
            if not isinstance(trans, Transform):
                raise TypeError("Chain needs to be a list of Transform objects!")

    def __call__(self, x):
        for trans in self.chain:
            x = trans(x)
        return x
