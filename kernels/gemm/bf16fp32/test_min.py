import torch
import tk_kernel

N = 8192
scale = 10.0

import torch
torch.manual_seed(0)
A = (torch.randn(N, N, dtype=torch.float32, device="cpu") / scale).to(torch.bfloat16).contiguous()
B = (torch.randn(N, N, dtype=torch.float32, device="cpu") / scale).to(torch.bfloat16).contiguous()
Bt = B.t().contiguous() # transpose B for the baseline gemm
C_torch = A@Bt

C_tk = torch.zeros((N, N), dtype=torch.bfloat16, device="cuda:0").contiguous()
tk_kernel.dispatch_micro(A.to("cuda:0"), B.to("cuda:0"), C_tk)

diff = (C_tk.cpu() - C_torch).abs()
tol = 1e-3 + 1e-2 * C_torch.abs()

bad = diff > tol

print("allclose:", not bad.any().item())
print("num bad:", bad.sum().item())
print("max diff:", diff.max().item())
print("max tol:", tol.max().item())
print("max violation:", (diff - tol).max().item())
