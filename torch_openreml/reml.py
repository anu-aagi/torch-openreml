import torch
from torch_openreml.utils import get_device, get_dtype
from torch_openreml.covariance.covariance_matrix import CovarianceMatrix
from tqdm import tqdm

class REML:
    
    def __init__(self, v_model=None, map_theta_to_v=None, map_theta_to_g=None, map_theta_to_dv=None):
        
        if v_model is not None and not isinstance(v_model, CovarianceMatrix):
            raise TypeError("'v_model' must be a CovarianceMatrix instance or None!")
        
        if v_model is None and map_theta_to_v is None:
            raise ValueError("At least one of 'v_model' or 'map_theta_to_v' must be provided!")
        
        if v_model is not None:
            self.map_theta_to_v = v_model.map_theta_to_v
            self.map_theta_to_dv = v_model.map_theta_to_dv
        else:
            self.map_theta_to_v = map_theta_to_v
            self.jacobian_func = torch.func.jacrev(map_theta_to_v)
            
        self.map_theta_to_g = map_theta_to_g
        self.map_theta_to_dv = map_theta_to_dv
      
    def blue(self, y, x, theta):
        device = get_device(y, x, theta)
        dtype = get_dtype(y, x, theta)
        
        scalar = {}
        matrix = {}
        
        scalar["N"] = y.shape[0]
        matrix["V"] = self.map_theta_to_v(theta)
        
        matrix["V"] = matrix["V"] + 1e-6 * torch.eye(scalar["N"], device=device, dtype=dtype)
        matrix["L"] = torch.linalg.cholesky(matrix["V"])
        
        matrix["Y"] = y.unsqueeze(-1)
        matrix["X"] = x
        
        matrix["V^{-1} Y"] = torch.cholesky_solve(matrix["Y"], matrix["L"])
        matrix["V^{-1} X"] = torch.cholesky_solve(matrix["X"], matrix["L"])
        
        matrix["X^T V^{-1} X"] = matrix["X"].T @ matrix["V^{-1} X"]
        matrix["X^T V^{-1} Y"] = matrix["X"].T @ matrix["V^{-1} Y"]
        
        matrix["L_{X^T V^{-1} X}"] = torch.linalg.cholesky(matrix["X^T V^{-1} X"])
        
        matrix[r"\hat{\beta}"] = torch.cholesky_solve(matrix["X^T V^{-1} Y"], matrix["L_{X^T V^{-1} X}"])
    
        return matrix[r"\hat{\beta}"].squeeze()
      
    def marginal_predict(self, y, x, theta):
        vector = {}
        
        vector[r"\hat{\beta}"] = self.blue(y, x, theta)
        return x @ vector[r"\hat{\beta}"]
    
    def marginal_residual(self, y, x, theta):
        return y - self.marginal_predict(y, x, theta)

    def blup(self, y, x, z, theta, map_theta_to_g=None):
        if map_theta_to_g is None:
            map_theta_to_g = self.map_theta_to_g
      
        device = get_device(y, x, z, theta)
        dtype = get_dtype(y, x, z, theta)
        
        scalar = {}
        matrix = {}
        
        scalar["N"] = y.shape[0]
        
        matrix["V"] = self.map_theta_to_v(theta)
        
        matrix["V"] = matrix["V"] + 1e-6 * torch.eye(scalar["N"], device=device, dtype=dtype)
        matrix["L"] = torch.linalg.cholesky(matrix["V"])
        
        matrix["Y"] = y.unsqueeze(-1)
        
        matrix["e"] = self.marginal_residual(y, x, theta).unsqueeze(-1)
        
        matrix["V^{-1} e"] = torch.cholesky_solve(matrix["e"], matrix["L"])
        
        matrix["G"] = map_theta_to_g(theta)
        
        return (matrix["G"] @ (z.T @ matrix["V^{-1} e"])).squeeze()
      
    def predict(self, y, x, z, theta, map_theta_to_g=None):
        if map_theta_to_g is None:
            map_theta_to_g = self.map_theta_to_g
            
        device = get_device(y, x, z, theta)
        dtype = get_dtype(y, x, z, theta)
        
        scalar = {}
        matrix = {}
        
        scalar["N"] = y.shape[0]
        matrix["V"] = self.map_theta_to_v(theta)
        
        matrix["V"] = matrix["V"] + 1e-6 * torch.eye(scalar["N"], device=device, dtype=dtype)
        matrix["L"] = torch.linalg.cholesky(matrix["V"])
        
        matrix["Y"] = y.unsqueeze(-1)
        matrix["X"] = x
        
        matrix["V^{-1} Y"] = torch.cholesky_solve(matrix["Y"], matrix["L"])
        matrix["V^{-1} X"] = torch.cholesky_solve(matrix["X"], matrix["L"])
        
        matrix["X^T V^{-1} X"] = matrix["X"].T @ matrix["V^{-1} X"]
        matrix["X^T V^{-1} Y"] = matrix["X"].T @ matrix["V^{-1} Y"]
        
        matrix["L_{X^T V^{-1} X}"] = torch.linalg.cholesky(matrix["X^T V^{-1} X"])
        
        matrix[r"\beta"] = torch.cholesky_solve(matrix["X^T V^{-1} Y"], matrix["L_{X^T V^{-1} X}"])
        
        matrix[r"\hat{Y}"] = matrix["X"] @ matrix[r"\beta"]
        
        matrix["e"] = matrix["Y"] - matrix[r"\hat{Y}"]
        
        matrix["V^{-1} e"] = torch.cholesky_solve(matrix["e"], matrix["L"])
        
        matrix["G"] = map_theta_to_g(theta)
        
        return (matrix[r"\hat{Y}"] + (matrix["G"] @ (z.T @ matrix["V^{-1} e"]))).squeeze()
      
    def residual(self, y, x, z, theta, map_theta_to_g=None):
        if map_theta_to_g is None:
            map_theta_to_g = self.map_theta_to_g
            
        return y - self.predict(y, x, z, theta, map_theta_to_g)
    
    def loglik(self, y, x, theta):
        device = get_device(y, x, theta)
        dtype = get_dtype(y, x, theta)
        
        scalar = {}
        matrix = {}
        
        scalar["N"] = y.shape[0]
        
        matrix["V"] = self.map_theta_to_v(theta)
        matrix["V"] = matrix["V"] + 1e-6 * torch.eye(n, device=device, dtype=dtype)
        
        matrix["L"] = torch.linalg.cholesky(matrix["V"])
        
        matrix["Y"] = y.unsqueeze(-1)
        
        matrix["V^{-1} Y"] = torch.cholesky_solve(matrix["Y"], matrix["L"])
        matrix["V^{-1} X"] = torch.cholesky_solve(matrix["X"], matrix["L"])
    
        matrix["X^T V^{-1} X"] = matrix["X"].T @ matrix["V^{-1} X"]
        matrix["X^T V^{-1} Y"] = matrix["X"].T @ matrix["V^{-1} Y"]
        
        matrix["L_{X^T V^{-1} X}"] = torch.linalg.cholesky(matrix["X^T V^{-1} X"])
        
        matrix[r"\hat{\beta}"] = torch.cholesky_solve(matrix["X^T V^{-1} Y"], matrix["L_{X^T V^{-1} X}"])
        
        matrix["e"] = matrix["Y"] - matrix["X"] @ matrix[r"\hat{\beta}"]
        
        matrix["V^{-1} e"] = torch.cholesky_solve(matrix["e"], matrix["L"])
        
        scalar["log |V|"] = 2.0 * torch.sum(torch.log(torch.diag(matrix["L"])))
        scalar["log |X^T V^{-1} X|"] = 2.0 * torch.sum(torch.log(torch.diag(matrix["L_{X^T V^{-1} X}"])))
        scalar["Y^T P Y"] = (matrix["e"].T @ matrix["V^{-1} e"]).squeeze()
        
        return -0.5 * (scalar["log |V|"] + scalar["log |X^T V^{-1} X|"] + scalar["Y^T P Y"])
          
    def compute_v_dv(self, theta):
        
        v = self.map_theta_to_v(theta)
        
        if self.map_theta_to_dv is None:
            jacobian = self.jacobian_func(theta)
            dv = jacobian.permute(2, 0, 1)
        else:
            dv = self.map_theta_to_dv(theta)
    
        return v, dv

    def ai_step(self, y, x, theta, require_loglik=True, require_beta=True):
        device = get_device(y, x, theta)
        dtype = get_dtype(y, x, theta)
        
        matrix = {}
        vector = {}
        scalar = {}
        tensor3d = {}
        
        matrix["X"] = x
        matrix["Y"] = y.unsqueeze(-1)
        scalar["N"] = y.shape[0]
        
        matrix["V"], tensor3d["dV"] = self.compute_v_dv(theta)
        scalar["K"] = len(tensor3d["dV"])
        
        matrix["V"] = matrix["V"] + 1e-6 * torch.eye(scalar["N"], device=device, dtype=dtype)
        
        matrix["L"] = torch.linalg.cholesky(matrix["V"])
        
        matrix["V^{-1} Y"] = torch.cholesky_solve(matrix["Y"], matrix["L"])
        matrix["V^{-1} X"] = torch.cholesky_solve(matrix["X"], matrix["L"])
        
        matrix["X^T V^{-1} X"] = matrix["X"].T @ matrix["V^{-1} X"]
        
        matrix["L_{X^T V^{-1} X}"] = torch.linalg.cholesky(matrix["X^T V^{-1} X"])
        
        matrix["(X^T V^{-1} X)^{-1} X^T V{-1}"] = torch.cholesky_solve(matrix["V^{-1} X"].T,
                                                                       matrix["L_{X^T V^{-1} X}"])
        
        matrix["V^{-1} X (X^T V^{-1} X)^{-1} X^T V{-1}"] = matrix["V^{-1} X"] @ matrix["(X^T V^{-1} X)^{-1} X^T V{-1}"]
        
        matrix["V^{-1} X (X^T V^{-1} X)^{-1} X^T V{-1} Y"] = matrix["V^{-1} X (X^T V^{-1} X)^{-1} X^T V{-1}"] @ matrix["Y"]
        
        matrix["P Y"] = matrix["V^{-1} Y"] - matrix["V^{-1} X (X^T V^{-1} X)^{-1} X^T V{-1} Y"]
        
        tensor3d["V^{-1} dV"] = torch.cholesky_solve(tensor3d["dV"], matrix["L"])
        
        tensor3d["V^{-1} X (X^T V^{-1} X)^{-1} X^T V{-1} dV"] = matrix["V^{-1} X (X^T V^{-1} X)^{-1} X^T V{-1}"] @ tensor3d["dV"]
        
        tensor3d["P dV"] = tensor3d["V^{-1} dV"] - tensor3d["V^{-1} X (X^T V^{-1} X)^{-1} X^T V{-1} dV"]
        
        tensor3d["P dV P Y"] = tensor3d["P dV"] @ matrix["P Y"]
        
        vector["Y^T P dV P Y"] = (matrix["Y"].T @ tensor3d["P dV P Y"]).squeeze()
        
        vector["tr(P dV)"] = torch.vmap(torch.trace)(tensor3d["P dV"])
        
        # Score vector
        vector["score"] = 0.5 * (vector["Y^T P dV P Y"] - vector["tr(P dV)"])
        
        # AI matrix
        tensor3d["Y^T P dV"] = matrix["Y"].T @ tensor3d["P dV"]
        
        matrix["AI"] = 0.5 * (tensor3d["Y^T P dV"].squeeze() @ tensor3d["P dV P Y"].squeeze().T)
                
        # REML log-likelihood
        if require_loglik:
            scalar["log |V|"] = 2.0 * torch.sum(torch.log(torch.diag(matrix["L"])))
            scalar["log |X^T V^{-1} X|"] = 2.0 * torch.sum(torch.log(torch.diag(matrix["L_{X^T V^{-1} X}"])))
            scalar["Y^T P Y"] = (matrix["Y"].T @ matrix["P Y"]).squeeze()
            scalar["loglik"] = -0.5 * (scalar["log |V|"] + scalar["log |X^T V^{-1} X|"] + scalar["Y^T P Y"])
        else:
            scalar["loglik"] = torch.nan
        
        # Beta
        if require_beta:
            vector["beta"] = (matrix["(X^T V^{-1} X)^{-1} X^T V{-1}"] @ matrix["Y"]).squeeze()
        else:
            vector["beta"] = torch.nan
    
        return vector["beta"], vector["score"], matrix["AI"], scalar["loglik"]
      
    def get_theta(self, select="last", history=None):
        if history is None:
            history = self.history
            
        if select == "last":
            return self.history["theta"][-1]
        else:
            if torch.is_tensor(self.history["loglik"][-1]):
                index = torch.argmax(torch.stack(self.history["loglik"])).item()
                return self.history["theta"][index]
            else:
                return self.history["theta"][-1]
              
    def get_beta(self, select="last", history=None):
        if history is None:
            history = self.history
            
        if select == "last":
            return self.history["beta"][-1]
        else:
            if torch.is_tensor(self.history["loglik"][-1]):
                index = torch.argmax(torch.stack(self.history["loglik"])).item()
                return self.history["beta"][index]
            else:
                return self.history["beta"][-1]
    
    def is_converged(self, 
                     check_score=True, 
                     check_delta=True,
                     check_loglik=True,
                     tol_score=1e-4, 
                     tol_delta=1e-4, 
                     tol_loglik=1e-4):
                       
        if len(self.history["score"]) < 2:
            return False
          
        if check_score:
            score_norm = torch.norm(self.history["score"][-1]).item()
            if score_norm >= tol_score:
                return False
        
        if check_delta:
            delta_norm = torch.norm(self.history["delta"][-1]).item()
            if delta_norm >= tol_delta:
                return False
              
        if check_loglik and torch.is_tensor(self.history["loglik"][-1]):
            loglik_diff = torch.abs(self.history["loglik"][-1] - self.history["loglik"][-2]).item()
            if loglik_diff >= tol_loglik:
                return False
        
        return True
          
    def update(self, theta, delta, eta, lb=-torch.inf, ub=torch.inf):
        last_theta = theta
        theta = theta + delta * eta
        theta = torch.clamp(theta, min=lb, max=ub)
        return theta, theta - last_theta
        
    def optimize(self, 
                 y, 
                 x, 
                 theta, 
                 max_iter=200, 
                 eta=0.5, 
                 require_loglik=True,
                 lb=-torch.inf, 
                 ub=torch.inf,
                 verbose=0, 
                 check_score=True,
                 check_delta=True,
                 check_loglik=True,
                 tol_score=1e-4, 
                 tol_delta=1e-4, 
                 tol_loglik=1e-4):
        self.history = {"theta": [], 
                        "beta": [], 
                        "loglik": [], 
                        "score": [], 
                        "ai": [], 
                        "delta": [],
                        "update": []}
        
        pb = tqdm(disable=not verbose, bar_format="{desc} \u23F1 {elapsed} | \u26A1 {rate_fmt}")
        
        with torch.no_grad():
            for i in range(max_iter):
                beta, score, ai, loglik = self.ai_step(y, x, theta, require_loglik=require_loglik)
                l_ai = torch.linalg.cholesky(ai)
                delta = torch.cholesky_solve(score.unsqueeze(-1), l_ai).squeeze()
                theta, update = self.update(theta, delta, eta, lb, ub)
                
                self.history["theta"].append(theta)
                self.history["beta"].append(beta)
                self.history["loglik"].append(loglik)
                self.history["score"].append(score)
                self.history["ai"].append(ai)
                self.history["delta"].append(delta)
                self.history["update"].append(update)
                
                if verbose > 0:
                    pb.set_description(f"Iter {i + 1}")
                    pb.update(1)
                    
                    if verbose > 1:
                        write_str = f"\u2225\u2207\u2225: {torch.norm(score):12.4f}, \u2225\u0394\u2225: {torch.norm(delta):6.4f}, \u03B7: {eta:.2f}, \u2225\u0394\u1D9C\u2225: {torch.norm(update):6.4f}"
                      
                        if require_loglik:
                            if len(self.history["loglik"]) > 1:
                                delta_loglik = self.history["loglik"][-1].item() - self.history["loglik"][-2].item()
                                write_str += f", log \U0001D4DB: {loglik:8.4f} ({delta_loglik:+.4f})"
                            else:
                                write_str += f", log \U0001D4DB: {loglik:8.4f}"
                        
                        if i == 0:
                            tqdm.write("")
                        tqdm.write(write_str)
                
                if self.is_converged(check_score, check_delta, check_loglik, tol_score, tol_delta, tol_loglik):
                    if verbose > 0:
                        if verbose > 1:
                            tqdm.write(f"\n[\u2207: score, \u0394: \U0001D409\u207B\u00B9\u2207, \u03B7: learning rate, \u0394\u1D9C: clip(\U0001D6C9 + \u03B7\u0394, lb, ub) - \U0001D6C9, \U0001D4DB: restricted likelihood]")
                        tqdm.write(f"\n\u2713 Converged at iteration {i + 1}")
                    break
        
        pb.close()
        
        return theta, beta, i + 1
  
