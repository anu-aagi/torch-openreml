import torch
import pandas as pd
  
class REML:
    
    def __init__(self, map_theta_to_v):
        self.map_theta_to_v = map_theta_to_v
        self.jacobian_func = torch.func.jacrev(map_theta_to_v)
    
    @staticmethod
    def get_device(*args):
        if len(args) == 0:
            return "cpu"
          
        device = args[0].device
        for i, t in enumerate(args):
            if t.device != device:
                raise ValueError(f"Device mismatch at arg {i}: {t.device} != {device}")
        
        return device
    
    @staticmethod
    def map_z_g_r_to_v(z, g, r):
        return z @ g @ z.T + r
      
    def blue(self, y, x, theta):
        device = REML.get_device(y, x, theta)
        
        n = y.shape[0]
        v = self.map_theta_to_v(theta)
        
        v = v + 1e-6 * torch.eye(n, device=device)
        l = torch.linalg.cholesky(v)
    
        y_mat = y.unsqueeze(-1)
    
        sinv_y = torch.cholesky_solve(y_mat, l)
        sinv_x = torch.cholesky_solve(x, l)
    
        xt = x.T
        xt_sinv_x = xt @ sinv_x
        xt_sinv_y = xt @ sinv_y
        return torch.linalg.solve(xt_sinv_x, xt_sinv_y).squeeze()
    
    def marginal_residual(self, y, x, theta):
        beta = self.blue(y, x, theta)
        y_mat = y.unsqueeze(-1)
        return y_mat - x @ beta

    def blup(self, y, x, z, theta, map_theta_to_g):
        device = REML.get_device(y, x, z, theta)
        
        n = y.shape[0]
    
        v = self.map_theta_to_v(theta)
        v = v + 1e-6 * torch.eye(n, device=device)
    
        l = torch.linalg.cholesky(v)
    
        r = self.marginal_residual(y, x, theta)
    
        sinv_res = torch.cholesky_solve(r, l)
    
        g = map_theta_to_g(theta)
    
        u = g @ (z.T @ sinv_res)
    
        return u.squeeze()
      
    def residual(self, y, x, z, theta, map_theta_to_g):
        device = REML.get_device(y, x, z, theta)
    
        beta = self.blue(y, x, theta)

        u = self.blup(y, x, z, theta, map_theta_to_g)

        y_mat = y.unsqueeze(-1)
    
        residual = y_mat - x @ beta - z @ u.unsqueeze(-1)
    
        return residual.squeeze()
    
    def loglik(self, y, x, theta):
        device = self.get_device(y, x, theta)
        n = y.shape[0]
        
        v = self.map_theta_to_v(theta)
        v = v + 1e-6 * torch.eye(n, device=device)
        l = torch.linalg.cholesky(v)
        
        y_mat = y.unsqueeze(-1)
    
        sinv_y = torch.cholesky_solve(y_mat, l)
        sinv_x = torch.cholesky_solve(x, l)
    
        xt = x.T
        xt_sinv_x = xt @ sinv_x
        xt_sinv_y = xt @ sinv_y
        
        beta = torch.linalg.solve(xt_sinv_x, xt_sinv_y)
        
        r = y_mat - x @ beta
        sinv_r = torch.cholesky_solve(r, l)
        
        logdet_v = 2.0 * torch.sum(torch.log(torch.diag(l)))
                
        c = torch.linalg.cholesky(xt_sinv_x)
        logdet_xt_vinv_x = 2.0 * torch.sum(torch.log(torch.diag(c)))
        
        quad = r.T @ sinv_r
        
        loglik = -0.5 * (
            logdet_v +
            logdet_xt_vinv_x +
            quad.squeeze()
        )
        
        return loglik
          
    def compute_v_dv(self, theta):
        jacobian = self.jacobian_func(theta)
        v = self.map_theta_to_v(theta)
        dv = [jacobian[..., i] for i in range(theta.shape[0])]
    
        return v, dv

    def ai_step(self, y, x, theta, require_loglik=True, require_beta=True):
        r"""
        Score:
        
        S_k = 0.5 * ((P Y)^T dV_k P Y - tr(P dV_k))
        
        AI:
        
        AI_{ij} = 0.5 * (P Y)^T dV_i P dV_j P Y
        
        P Y:
        
        P Y = V^{-1} Y - V^{-1} X (X^T V^{-1} X)^{-1} X^T V^{-1} Y
        
        P dV_k:
        
        P dV_k = V^{-1} dV_k - V^{-1} X (X^T V^{-1} X)^{-1} X^T V^{-1} dV_k
        """
        
        device = self.get_device(y, x, theta)
        
        matrix = {}
        vector = {}
        scalar = {}
        matrix_list = {}
        
        matrix["X"] = x
        matrix["Y"] = y.unsqueeze(-1)
        scalar["N"] = y.shape[0]
        
        matrix["V"], matrix_list["dV"] = self.compute_v_dv(theta)
        scalar["K"] = len(matrix_list["dV"])
        
        matrix["V"] = matrix["V"] + 1e-6 * torch.eye(scalar["N"], device=device)
        
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
            
        if require_beta:
            vector["beta"] = (matrix["(X^T V^{-1} X)^{-1} X^T V{-1}"] @ matrix["Y"]).squeeze()
        else:
            vector["beta"] = torch.nan
    
        return vector["beta"], vector["score"], matrix["AI"], scalar["loglik"]
    
    def is_converged(self, tol_score=1e-4, tol_delta=1e-4):
        if len(self.history["score"]) < 2:
            return False
    
        score_norm = torch.norm(self.history["score"][-1]).item()
        delta_norm = torch.norm(self.history["delta"][-1]).item()
    
        return score_norm < tol_score and delta_norm < tol_delta
    
    def optimize(self, y, x, theta, max_iter=50, lr=0.5, require_loglik=True):
        device = self.get_device(y, x, theta)
        
        self.history = {"theta": [], 
                        "beta": [], 
                        "loglik": [], 
                        "score": [], 
                        "ai": [], 
                        "delta": []}
        
        with torch.no_grad():
            for i in range(max_iter):
                beta, score, ai, loglik = self.ai_step(y, x, theta, require_loglik=require_loglik)
                delta = torch.linalg.solve(ai, score)
                theta = theta + lr * delta
                
                self.history["theta"].append(theta)
                self.history["beta"].append(beta)
                self.history["loglik"].append(loglik)
                self.history["score"].append(score)
                self.history["ai"].append(ai)
                self.history["delta"].append(delta)
                
                if self.is_converged():
                  break
        
        return theta, beta, i
      
if __name__ == "__main__":
    
    # -----------------------------
    # Load Data
    # -----------------------------
    data = [
    (1,"R1","B1","G11",4.1172),(2,"R1","B1","G04",4.4461),(3,"R1","B1","G05",5.8757),
    (4,"R1","B1","G22",4.5784),(5,"R1","B2","G21",4.6540),(6,"R1","B2","G10",4.1736),
    (7,"R1","B2","G20",4.0141),(8,"R1","B2","G02",4.3350),(9,"R1","B3","G23",4.2323),
    (10,"R1","B3","G14",4.7572),(11,"R1","B3","G16",4.4906),(12,"R1","B3","G18",3.9737),
    (13,"R1","B4","G13",4.2530),(14,"R1","B4","G03",3.3420),(15,"R1","B4","G19",4.7269),
    (16,"R1","B4","G08",4.9989),(17,"R1","B5","G17",4.7876),(18,"R1","B5","G15",5.0902),
    (19,"R1","B5","G07",4.1505),(20,"R1","B5","G01",5.1202),(21,"R1","B6","G06",4.7085),
    (22,"R1","B6","G12",5.2560),(23,"R1","B6","G24",4.9577),(24,"R1","B6","G09",3.3986),
    (25,"R2","B1","G08",3.9926),(26,"R2","B1","G20",3.6056),(27,"R2","B1","G14",4.5294),
    (28,"R2","B1","G04",4.3599),(29,"R2","B2","G24",3.9039),(30,"R2","B2","G15",4.9114),
    (31,"R2","B2","G03",3.7999),(32,"R2","B2","G23",4.3042),(33,"R2","B3","G12",5.3127),
    (34,"R2","B3","G11",5.1163),(35,"R2","B3","G21",5.3802),(36,"R2","B3","G17",5.0744),
    (37,"R2","B4","G05",5.1202),(38,"R2","B4","G09",4.2955),(39,"R2","B4","G10",4.9057),
    (40,"R2","B4","G01",5.7161),(41,"R2","B5","G02",5.1566),(42,"R2","B5","G18",5.0988),
    (43,"R2","B5","G13",5.4840),(44,"R2","B5","G22",5.0969),(45,"R2","B6","G19",5.3148),
    (46,"R2","B6","G07",4.6297),(47,"R2","B6","G06",5.1751),(48,"R2","B6","G16",5.3024),
    (49,"R3","B1","G11",3.9205),(50,"R3","B1","G01",4.6512),(51,"R3","B1","G14",4.3887),
    (52,"R3","B1","G19",4.5552),(53,"R3","B2","G02",4.0510),(54,"R3","B2","G15",4.6783),
    (55,"R3","B2","G09",3.1407),(56,"R3","B2","G08",3.9821),(57,"R3","B3","G17",4.3234),
    (58,"R3","B3","G18",4.2486),(59,"R3","B3","G04",4.3960),(60,"R3","B3","G06",4.2474),
    (61,"R3","B4","G12",4.1746),(62,"R3","B4","G13",4.7512),(63,"R3","B4","G10",4.0875),
    (64,"R3","B4","G23",3.8721),(65,"R3","B5","G21",4.4130),(66,"R3","B5","G22",4.2397),
    (67,"R3","B5","G16",4.3852),(68,"R3","B5","G24",3.5655),(69,"R3","B6","G03",2.8873),
    (70,"R3","B6","G05",4.1972),(71,"R3","B6","G20",3.7349),(72,"R3","B6","G07",3.6096),
    ]
    
    df = pd.DataFrame(data, columns=["plot","rep","block","gen","yield"])
    
    # -----------------------------
    # Design matrices
    # -----------------------------
    x = torch.tensor(pd.get_dummies(df["rep"]).values, dtype=torch.float32)
    z_gen = torch.tensor(pd.get_dummies(df["gen"]).values, dtype=torch.float32)
    
    df["rep_block"] = df["rep"] + ":" + df["block"]
    z_rb = torch.tensor(pd.get_dummies(df["rep_block"]).values, dtype=torch.float32)
    
    y = torch.tensor(df["yield"].values, dtype=torch.float32)
    n = len(y)
    
    k_gen = z_gen @ z_gen.T
    k_rb = z_rb @ z_rb.T
    i_n = torch.eye(n)
    
    # User spcify mapping function
    def map_theta_to_v(theta):
        if not (theta.device == z_gen.device == z_rb.device):
            raise ValueError(
                f"Device mismatch: "
                f"theta={theta.device}, "
                f"z_gen={z_gen.device}, "
                f"z_rb={z_rb.device}"
            )
      
        sigma2 = torch.exp(theta) ** 2
        v = k_gen * sigma2[0] + k_rb * sigma2[1] + sigma2[2] * i_n
                
        return v
      
    mod = REML(map_theta_to_v)
    mod.optimize(y, x, torch.tensor([0.0, 0.0, 0.0]), require_loglik=True, max_iter=50)
    
