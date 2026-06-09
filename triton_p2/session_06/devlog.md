# Session 06 Devlog — LayerNorm

## What LayerNorm does
y = (x - mean) / sqrt(variance + eps) * weight + bias

1. Compute mean of the row
2. Compute variance of the row
3. Normalize: subtract mean, divide by sqrt(variance + eps)
4. Scale by weight (learned, one value per feature)
5. Shift by bias (learned, one value per feature)

## eps
- Small constant (1e-5) added to variance before sqrt
- Prevents division by zero when variance is 0

## weight and bias
- Learned parameters, same shape as one row (one value per feature)
- Applied elementwise: multiply by weight, add bias
- x has 512 features → weight has 512 values, bias has 512 values
- Element 0 gets weight[0] and bias[0], element 1 gets weight[1] and bias[1]

## Why LayerNorm matters
- In every transformer layer — stabilizes training, controls activation scale
- Used in GPT, BERT, LLaMA (though LLaMA uses RMSNorm instead)
