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
