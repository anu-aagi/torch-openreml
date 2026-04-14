import torch
from torch_openreml.utils import get_device, get_dtype
from tqdm import tqdm

class REML:
    
    def __init__(self, map_theta_to_v, map_theta_to_g=None, map_theta_to_dv=None):
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
        matrix_list = {}
        
        matrix["X"] = x
        matrix["Y"] = y.unsqueeze(-1)
        scalar["N"] = y.shape[0]
        
        matrix["V"], matrix_list["dV"] = self.compute_v_dv(theta)
        scalar["K"] = len(matrix_list["dV"])
        
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
        
        matrix_list["V^{-1} dV_k"] = [torch.cholesky_solve(dv_k, matrix["L"]) for dv_k in matrix_list["dV"]]
        
        matrix_list["V^{-1} X (X^T V^{-1} X)^{-1} X^T V{-1} dV_k"] = [matrix["V^{-1} X (X^T V^{-1} X)^{-1} X^T V{-1}"] @ dv_k for dv_k in matrix_list["dV"]]
        
        matrix_list["P dV_k"] = [matrix_list["V^{-1} dV_k"][i] - matrix_list["V^{-1} X (X^T V^{-1} X)^{-1} X^T V{-1} dV_k"][i] for i in range(scalar["K"])]
        
        matrix_list["P dV_k P Y"] = [p_dv_k @ matrix["P Y"] for p_dv_k in matrix_list["P dV_k"]]
        
        matrix["P dV P Y"] = torch.cat(matrix_list["P dV_k P Y"], dim=1)
        vector["Y^T P dV P Y"] = (matrix["Y"].T @ matrix["P dV P Y"]).squeeze()
        
        vector["tr(P dV)"] = torch.stack([torch.trace(p_dv_k) for p_dv_k in matrix_list["P dV_k"]])
        
        # Score vector
        vector["score"] = 0.5 * (vector["Y^T P dV P Y"] - vector["tr(P dV)"])
        
        # AI matrix
        matrix["AI"] = torch.ones(scalar["K"], scalar["K"])
        
        for k in range(scalar["K"]):
            for j in range(scalar["K"]):
                if k > j:
                    next
                entry = 0.5 * (matrix["Y"].T @ matrix_list["P dV_k"][k] @ matrix_list["P dV_k P Y"][j])
                matrix["AI"][k][j] = entry
                matrix["AI"][j][k] = entry
                
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
    
    def is_converged(self, tol_score=1e-4, tol_delta=1e-4, tol_loglik=1e-4):
        if len(self.history["score"]) < 2:
            return False
    
        score_norm = torch.norm(self.history["score"][-1]).item()
        delta_norm = torch.norm(self.history["delta"][-1]).item()
        
        if torch.is_tensor(self.history["loglik"][-1]):
            loglik_diff = torch.abs(self.history["loglik"][-1] - self.history["loglik"][-2]).item()
            return score_norm < tol_score and delta_norm < tol_delta and loglik_diff < tol_loglik
        else:
            return score_norm < tol_score and delta_norm < tol_delta
        
    def optimize(self, y, x, theta, max_iter=200, eta=0.5, require_loglik=True, verbose=0, tol_score=1e-4, tol_delta=1e-4, tol_loglik=1e-4):
        self.history = {"theta": [], 
                        "beta": [], 
                        "loglik": [], 
                        "score": [], 
                        "ai": [], 
                        "delta": []}
        
        pb = tqdm(disable=not verbose, bar_format="{desc} \u23F1 {elapsed} | \u26A1 {rate_fmt}")
        
        with torch.no_grad():
            for i in range(max_iter):
                beta, score, ai, loglik = self.ai_step(y, x, theta, require_loglik=require_loglik)
                l_ai = torch.linalg.cholesky(ai)
                delta = torch.cholesky_solve(score.unsqueeze(-1), l_ai).squeeze()
                theta = theta + eta * delta
                
                self.history["theta"].append(theta)
                self.history["beta"].append(beta)
                self.history["loglik"].append(loglik)
                self.history["score"].append(score)
                self.history["ai"].append(ai)
                self.history["delta"].append(delta)
                
                if verbose > 0:
                    pb.set_description(f"Iter {i + 1}")
                    pb.update(1)
                    
                    if verbose > 1:
                        write_str = f"\u2225\u2207\u2225: {torch.norm(score):12.4f}, \u2225\u0394\u2225: {torch.norm(delta):6.4f}, \u03B7: {eta:.2f}"
                      
                        if require_loglik:
                            if len(self.history["loglik"]) > 1:
                                delta_loglik = self.history["loglik"][-1].item() - self.history["loglik"][-2].item()
                                write_str += f", log \U0001D4DB: {loglik:8.4f} ({delta_loglik:+.4f})"
                            else:
                                write_str += f", log \U0001D4DB: {loglik:8.4f}"
                        
                        tqdm.write(write_str)
                
                if self.is_converged(tol_score, tol_delta, tol_loglik):
                    if verbose:
                        tqdm.write(f"\n\u2713 Converged at iteration {i + 1}")
                    break
        
        pb.close()
        
        return theta, beta, i + 1
  
