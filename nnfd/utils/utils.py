import os
import torch
import torch.distributed as dist
from typing import Sequence

try:
    import gpustat
    _GPUSTAT_AVAILABLE = True
except ImportError:
    _GPUSTAT_AVAILABLE = False


def printc(text, color=None):
    if color == 'r':
        print("\033[91m" + text + "\033[0m", flush=True)
    elif color == 'g':
        print("\033[92m" + text + "\033[0m", flush=True)
    elif color == 'y':
        print("\033[93m" + text + "\033[0m", flush=True)
    elif color == 'b':
        print("\033[94m" + text + "\033[0m", flush=True)
    else:
        print(text, flush=True)

def MAE(a, b):
    return torch.mean(torch.abs(a-b))/torch.mean(torch.abs(b))

def scaled_MAE(a, b):
    scaled_a = a / torch.norm(a)
    scaled_b = b / torch.norm(b)
    diff = a - b
    return torch.mean(torch.abs(scaled_a-scaled_b))/torch.mean(torch.abs(scaled_b)), diff

def c2r(x):
    bs, sx, sy, sz, _ = x.shape
    return torch.view_as_real(x).reshape(bs, sx, sy, sz, 6)

def r2c(x):
    bs, sx, sy, sz, _ = x.shape
    return torch.view_as_complex(x.reshape(bs, sx, sy, sz, 3, 2))

def is_array_like(x):
    if isinstance(x, (str, bytes)):
        return False
    return isinstance(x, Sequence) or hasattr(x, "__array__") or torch.is_tensor(x)

def is_multiple(a, b, tol=1e-9):
    if b == 0:
        return False  # avoid division by zero
    if is_array_like(a):
        for i in range(len(a)):
            if not is_multiple(a[i], b, tol):
                return False
        return True
    else:
        quotient = a / b
        return abs(round(quotient) - quotient) < tol

def resolve(a, b, tol=1e-9):
    assert is_multiple(a, b, tol=tol)
    return round(a / b)

class IdentityModel:
    def setup(self, x, freq):
        pass
    def __call__(self, x, freq):
        return x

def init_dist():
    if dist.is_initialized():
        return

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '8778'

    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=1,
        rank=0
    )

def get_least_used_gpu():
    if not _GPUSTAT_AVAILABLE:
        raise RuntimeError("gpustat is not installed. Run: pip install gpustat")
    stats = gpustat.GPUStatCollection.new_query()
    ids = map(lambda gpu: int(gpu.entry['index']), stats)
    ratios = map(lambda gpu: float(gpu.entry['memory.used'])/float(gpu.entry['memory.total']), stats)
    bestGPU = min(zip(ids, ratios), key=lambda x: x[1])[0]
    return bestGPU

def get_pixels(kwargs, key, dL):
    value = kwargs[key]
    if value % dL != 0:
        printc(f"Warning: {key} is not divisible by dL, rounding to the nearest integer", 'r')
    return round(value / dL)

def smooth_edges(data, window=5):
    assert len(data.shape) == 2
    sx, sy = data.shape

    edge = torch.linspace(-int(window/2),int(window/2),window)
    edge = torch.sigmoid(edge)

    smooth_x = torch.ones(sx)
    smooth_x[:window] = edge
    smooth_x[-window:] = torch.flip(edge,dims=(0,))

    smooth_y = torch.ones(sy)
    smooth_y[:window] = edge
    smooth_y[-window:] = torch.flip(edge,dims=(0,))

    data = data * smooth_x[:,None]
    data = data * smooth_y[None,:]

    return data
