# make venv and pip install torch and numpy iirc
import torch
import torch.nn.functional as F

print("silu 10 : ", F.silu(torch.tensor([10.0])).item())
print("gelu 1 : ", F.gelu(torch.tensor([1.0])).item())
print("gelu -1 : ", F.gelu(torch.tensor([-1.0])).item())
