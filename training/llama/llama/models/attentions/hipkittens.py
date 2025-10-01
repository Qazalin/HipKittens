import torch
import torch.nn as nn
from torch.autograd import Function

# import tk_kernel_fwd
# import tk_kernel_bkwd

class HipAttnFunction(Function):
    """
    Inputs/outputs are BNHD (batch, seq, heads, dim), like your harness.
    Forward:  O, L  via tk_kernel_fwd.dispatch_fwd
    Backward: dQ,dK,dV via tk_kernel_bkwd.{dispatch_prep,dispatch_bwd_combined,dispatch_dq_shuffle}
    Compute in bf16, save L and O for backward, return O in input dtype.
    """

    @staticmethod
    def forward(ctx, q_bnhd: torch.Tensor, k_bnhd: torch.Tensor, v_bnhd: torch.Tensor):
        B, N, H, D = q_bnhd.shape
        pass

    @staticmethod
    def backward(ctx, dO_bnhd: torch.Tensor):
        q, k, v, O, L = ctx.saved_tensors
        pass


class HipSelfAttention(nn.Module):
    """Implement the scaled dot product attention with softmax.
    Arguments
    ---------
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.0)
    """

    def __init__(
        self,
        causal=False,
        softmax_scale=None,
        attention_dropout=0.0,
        deterministic=False,
    ):
        super().__init__()
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.drop = nn.Dropout(attention_dropout)
        self.deterministic = deterministic

    def forward(self, qkv, causal=None, key_padding_mask=None):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            qkv: The tensor containing the query, key, and value. (B, S, 3, H, D)
            causal: if passed, will override self.causal
            key_padding_mask: boolean mask to apply to the attention weights. True means to keep,
                False means to mask out. (B, S)
        """
        assert qkv.dtype in [torch.float16, torch.bfloat16]
        assert qkv.is_cuda
        causal = self.causal if causal is None else causal
        q, k, v = qkv.unbind(dim=2)
        out_bnhd = HipAttnFunction.apply(q, k, v)  
        ctx = out_bnhd.to(q.dtype).contiguous()
        return ctx
        

