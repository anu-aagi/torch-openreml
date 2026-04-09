def map_z_g_r_to_v(z, g, r):
    return z @ g @ z.T + r
  
def get_device(*args):
    if len(args) == 0:
        return "cpu"
      
    device = args[0].device
    for i, t in enumerate(args):
        if t.device != device:
            raise ValueError(f"Device mismatch at arg {i}: {t.device} != {device}")
    
    return device

def get_dtype(*args):
    if len(args) == 0:
        return torch.get_default_dtype()
    
    dtype = args[0].dtype
    for i, t in enumerate(args):
        if t.dtype != dtype:
            raise ValueError(f"Dtype mismatch at arg {i}: {t.dtype} != {dtype}")
    
    return dtype
