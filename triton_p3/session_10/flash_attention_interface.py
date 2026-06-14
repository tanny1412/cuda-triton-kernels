import torch
import torch.nn as nn
import triton
from flash_attention_causal import flash_attention_causal


class FlashAttentionFunction(torch.autograd.Function):
    """Low-level wrapper — registers the Triton kernel with PyTorch autograd."""

    @staticmethod
    def forward(ctx, Q, K, V):
        O = flash_attention_causal(Q, K, V)
        ctx.save_for_backward(Q, K, V, O)
        return O

    @staticmethod
    def backward(ctx, dO):
        # inference only — backward not implemented
        # a full implementation would compute dQ, dK, dV here
        return None, None, None


class FlashAttentionLayer(nn.Module):
    """Drop-in nn.Module — use this inside transformer models."""

    def forward(self, Q, K, V):
        return FlashAttentionFunction.apply(Q, K, V)


# test — works like any PyTorch op
N, D = 512, 64
Q = torch.randn(N, D, device='cuda')
K = torch.randn(N, D, device='cuda')
V = torch.randn(N, D, device='cuda')

layer = FlashAttentionLayer()
O = layer(Q, K, V)
print(f"output shape: {O.shape}")
print(f"O[0][0] = {O[0][0].item():.4f}")

# works with torch.compile
compiled_layer = torch.compile(layer)
O_compiled = compiled_layer(Q, K, V)
print(f"torch.compile output matches: {torch.allclose(O, O_compiled, atol=1e-3)}")
