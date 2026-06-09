# Session 02 Devlog — Memory Access + Masks

## Masked load default value
- tl.load with mask=False elements → get value from `other` parameter
- Default other=0.0 if not specified
- tl.load(ptr + offsets, mask=mask, other=-1.0) → masked elements = -1.0
- Use other=0.0 for sums, other=float('-inf') for softmax

## Mask on both load and store
- tl.load  with mask → only load valid elements, masked get `other` value
- tl.store with mask → only write valid elements, masked positions untouched
- Invalid elements: loaded as -1 → computed as -3.0 → never written to output
- Output tensor is never corrupted by out-of-bounds elements

## Partial last block example (N=100, BLOCK_SIZE=64)
- Block 0: elements 0-63   → full block, all valid
- Block 1: elements 64-127 → partial, only 64-99 valid (36 elements), 100-127 masked
- Without mask: reads garbage memory, stores garbage to output
- With mask: only 36 valid elements loaded and stored, rest ignored

## Why other value matters for debugging
- Setting other=-1.0 makes masked elements visible during debugging
- If you accidentally remove mask from tl.store, you'd see -3.0 in output
- In production: use mathematically correct values (0.0, -inf, etc.)

## Scalar arguments
- Regular Python scalars (int, float) can be passed directly to Triton kernels
- They flow from Python → host function → kernel argument → GPU computation
- No special handling needed unlike tensors (which need device='cuda')
