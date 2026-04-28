import torch

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

def numeric_to_design_matrix(*args, dtype=None, device=None):

    if len(args) == 0:
        raise ValueError("At least one input is required.")

    cols = []
    n = None

    dtype = dtype or torch.float32
    device = device or "cpu"

    for i, x in enumerate(args):

        if torch.is_tensor(x):
            x = x.to(dtype=dtype, device=device)
        elif isinstance(x, (list, tuple)):
            x = torch.tensor(x, dtype=dtype, device=device)
        else:
            raise TypeError("'x' must be either a tensor or a list/tuple of values!")

        if len(x.shape) == 2:
            x.unsqueeze_(0)

        if n is None:
            n = x.shape[0]
        elif x.shape[0] != n:
            raise ValueError(f"Inconsistent lengths at argument {i}: expected {n}, got {x.shape[0]}")

        cols.append(x)

    return torch.stack(cols, dim=1)

def categorical_to_design_matrix(x, levels=None, drop_first=False, dtype=None, device=None):

    dtype = dtype or torch.float32
    device = device or "cpu"

    if levels is None:
        levels = sorted(set(x))

    if len(levels) != len(set(x)):
        raise ValueError("'levels' must match the number of unique values in 'x'!")

    level_to_idx = {lev: i for i, lev in enumerate(levels)}

    idx = torch.tensor([level_to_idx[v] for v in x], device=device, dtype=torch.long)

    z = torch.nn.functional.one_hot(idx, num_classes=len(levels)).to(dtype=dtype)

    if drop_first:
        z = z[:, 1:]

    return z

def augment(*args):
    return torch.cat(args, dim=1)

def interaction(*args, sep="\u22C8"):
    if len(args) == 0:
        raise ValueError("At least one input is required.")

    for i, arg in enumerate(args):
        if not isinstance(arg, (list, tuple)):
            raise TypeError(f"Argument {i} is not list/tuple of strings!")

    return [sep.join(parts) for parts in zip(*args)]