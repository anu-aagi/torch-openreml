from torch_openreml.covariance.covariance_matrix import CovarianceMatrix, OperatorMatrix
import torch

class SumMatrix(OperatorMatrix):
    
    def __init__(self, operands):
        if (len(operands) < 2):
            raise ValueError("At least two operands are required")
          
        n = next(
            operand.n 
            for operand in operands.values() 
            if isinstance(operand, CovarianceMatrix)
        )
          
        super().__init__(n, operands)
        
    def build(self, params, grad=True):
        params = self.from_param_dict(params)
        self.check_params(params)
        
        v = None
        
        if grad:
            self.reset_grad()
            self._grad = []
        
        for name, operand in self.operands.items():
            if isinstance(operand, CovarianceMatrix):
                operand_params = params[0:operand.num_params]
                params = params[operand.num_params:]
                
                if v is None:
                    v = operand.build(operand_params)
                else:
                    v = v + operand.build(operand_params)
                    
                if grad:
                    if operand.grad is not None:
                        self._grad.append(operand.grad)
                        self._grad_names.extend(operand.grad_names)
        
        if grad:
            if len(self._grad) == 0:
                self._grad = None
            else:
                self._grad = torch.cat(self._grad)
                    
        return v
