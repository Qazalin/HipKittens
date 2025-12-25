import torch
import tk_kernel
from aiter.tuned_gemm import tgemm

N = 8192
scale = 10.0
gpu = "cuda:0"

import torch
torch.manual_seed(0)
A = (torch.randn(N, N, dtype=torch.float32, device="cpu") / scale).to(torch.bfloat16).contiguous()
B = (torch.randn(N, N, dtype=torch.float32, device="cpu") / scale).to(torch.bfloat16).contiguous()
Bt = B.t().contiguous() # transpose B for the baseline gemm
C_torch = A@Bt

C_aiter = tgemm.mm(A.to(gpu), B.to(gpu), None, None, None)
assert torch.allclose(C_aiter.to("cpu"), C_torch, rtol=1e-2, atol=1e-3)

C_tk = torch.zeros((N, N), dtype=torch.bfloat16, device=gpu).contiguous()
tk_kernel.dispatch_micro(A.to(gpu), B.to(gpu), C_tk)

diff = (C_tk.cpu() - C_torch).abs()
tol = 1e-3 + 1e-2 * C_torch.abs()

bad = diff > tol

print("allclose:", not bad.any().item())
print("num bad:", bad.sum().item())
print("max diff:", diff.max().item())
print("max tol:", tol.max().item())
print("max violation:", (diff - tol).max().item())
