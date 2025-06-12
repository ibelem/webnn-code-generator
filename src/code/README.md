# WebNN and ONNX Operator Mapping

This comprehensive mapping document lists all WebNN operators with their corresponding ONNX operators, organized alphabetically by category. It includes all operations found in the webnn-code-generator project across all subdirectories.

## Table of Contents
- [Activation](#activation)
- [Attention](#attention)
- [Convolution](#convolution)
- [Elementwise Binary](#elementwise-binary)
- [Elementwise Logical](#elementwise-logical)
- [Elementwise Unary](#elementwise-unary)
- [Elementwise Others](#elementwise-others)
- [Linear](#linear)
- [Normalization](#normalization)
- [Pooling](#pooling)
- [Quantization](#quantization)
- [Reduction](#reduction)
- [RNN](#rnn)
- [Shape](#shape)
- [Tensor](#tensor)
- [Misc](#misc)

## Activation (12)

| # | WebNN Op | ONNX Op | Note |
|---|----------|---------|------|
| 1 | elu | Elu | `alpha` option supported |
| 2 | gelu | Gelu | Default implementation uses exact calculation |
| 3 | hardSigmoid | HardSigmoid | `alpha` and `beta` options supported |
| 4 | hardSwish | HardSwish | Direct mapping |
| 5 | leakyRelu | LeakyRelu | `alpha` option supported |
| 6 | prelu | PRelu | Second input is slope tensor |
| 7 | relu | Relu | Direct mapping |
| 8 | sigmoid | Sigmoid | Direct mapping |
| 9 | softmax | Softmax | `axis` option supported |
| 10 | softplus | Softplus | Direct mapping |
| 11 | softsign | Softsign | Direct mapping |
| 12 | tanh | Tanh | Direct mapping |

## Attention (0)

| # | WebNN Op | ONNX Op | Note |
|---|----------|---------|------|
|| N/A | GroupQueryAttention | Decomposed to WebNN subgraph |
|| N/A | MultiHeadAttention | Decomposed to WebNN subgraph |
|| N/A | RotaryEmbedding | Decomposed to WebNN subgraph |
|| N/A | ScaledDotProductAttention | Helper function for attention ops |

## Convolution (2)

| # | WebNN Op | ONNX Op | Note |
|---|----------|---------|------|
| 1 | conv2d | Conv | Support for NCHW/NHWC layouts |
| | conv2d | DepthwiseConv2d | Implemented using groups parameter |
| | conv2d | ConvInteger | Decomposed |
| 2 | convTranspose2d | ConvTranspose | |

## Elementwise Binary (7)

| # | WebNN Op | ONNX Op | Note |
|---|----------|---------|------|
| 1 | add | Add | Addition of two tensors |
| 2 | div | Div | Division of two tensors |
| 3 | max | Max | Element-wise maximum of two tensors |
| 4 | min | Min | Element-wise minimum of two tensors |
| 5 | mul | Mul | Multiplication of two tensors |
| 6 | pow | Pow | Element-wise power operation |
| 7 | sub | Sub | Subtraction of two tensors |

## Elementwise Logical (10)

| # | WebNN Op | ONNX Op | Note |
|---|----------|---------|------|
| 1 | equal | Equal | Equality comparison, returns boolean tensor |
| 2 | greater | Greater | Greater than comparison |
| 3 | greaterOrEqual | GreaterOrEqual | Greater than or equal comparison |
| 4 | less | Less | Less than comparison |
| 5 | lessOrEqual | LessOrEqual | Less than or equal comparison |
| 6 | logicalAnd | And | Logical AND operation |
| 7 | logicalNot | Not | Logical NOT operation |
| 8 | logicalOr | Or | Logical OR operation |
| 9 | logicalXor | Xor | Logical XOR operation |
| 10 | notEqual | 🚧 | Not equal operation |

## Elementwise Unary (16)

| # | WebNN Op | ONNX Op | Note |
|---|----------|---------|------|
| 1 | abs | Abs | Absolute value |
| 2 | ceil | Ceil | Ceiling function |
| 4 | cos | Cos | Cosine function |
| 5 | erf | Erf | Error function |
| 6 | exp | Exp | Exponential function |
| 7 | floor | Floor | Floor function |
| 8 | identity | Identity | Passes through input unchanged |
| 9 | log | Log | Natural logarithm |
| 10 | neg | Neg | Negation (multiply by -1) |
| 11 | reciprocal | Reciprocal | 1/x operation |
| 12  | round 🚧 | Round | Rounds to nearest integer |
| 13 | sign | Sign | Returns sign of the input |
| 14 | sin | Sin | Sine function |
| 15 | sqrt | Sqrt | Square root function |
| 16 | tan | Tan | Tangent function |

## Elementwise Others (6)

| # | WebNN Op | ONNX Op | Note |
|---|----------|---------|------|
| 1 | clamp | Clip | Clamps values between min and max |
| 2 | expand | Expand | Broadcasts tensor to new shape  |
| 3 | pad | Pad | Supports constant, reflection, and edge modes |
| 4 | slice | Slice | Extracts slice with optional steps |
| 5 | tile | Tile | Repeats tensor along dimensions |
| 6 | where | Ternary | |

## Linear (3)

| # | WebNN Op | ONNX Op | Note |
|---|----------|---------|------|
| 1 | gemm | Gemm | General matrix multiplication with bias |
| 2 | matmul | MatMul | Matrix multiplication |
| | N/A | MatMulInteger 🚧 | Quantized matmul, decomposed |
| | N/A | MatMulNBits | Decomposed to dequantizeLinear + matmul |
| 3 | triangular | Triangular | Upper or lower triangular matrix |

## Normalization (3)

| # | WebNN Op | ONNX Op | Note |
|---|----------|---------|------|
| 1 | batchNormalization | BatchNormalization | `axis` option set based on layout |
| 2 | instanceNormalization | InstanceNormalization | `layout` and `epsilon` options supported |
| 3 | layerNormalization | LayerNormalization | Supports axes and epsilon options |
| | N/A | LRN | Decomposed to multiple WebNN ops |
| | N/A | SimplifiedLayerNormalization | Decomposed to WebNN ops |
| | N/A | SkipLayerNormalization 🚧 | Decomposed to WebNN ops |
| | N/A | SkipSimplifiedLayerNormalization | Decomposed to WebNN ops |

## Pooling (3)

| # | WebNN Op | ONNX Op | Note |
|---|----------|---------|------|
| 1 | averagePool2d | AveragePool | 2D only, layout sensitive |
| | averagePool2d | GlobalAveragePool | Uses input shape as window dimensions |
| 2 | l2Pool2d | GlobalLpPool | Only p=2 supported |
| | l2Pool2d | LpPool | Only p=2 supported |
| 3 | maxPool2d | GlobalMaxPool | Uses input shape as window dimensions |
| | maxPool2d | MaxPool | 2D only, layout sensitive |

## Quantization (2)

| # | WebNN Op | ONNX Op | Note |
|---|----------|---------|------|
| 1 | dequantizeLinear | DequantizeLinear | Converts quantized to float |
| 2 | quantizeLinear | QuantizeLinear | Converts float to quantized |

## Reduction (12)

| # | WebNN Op | ONNX Op | Note |
|---|----------|---------|------|
| 1 | reduceL1 | ReduceL1 | L1 norm reduction |
| 2 | reduceL2 | ReduceL2 | L2 norm reduction |
| 3 | reduceLogSum | ReduceLogSum | Log of sum reduction |
| 4 | reduceLogSumExp | ReduceLogSumExp | Log of sum of exponents reduction |
| 5 | reduceMax | ReduceMax | Maximum value reduction |
| 6 | reduceMean | ReduceMean | Mean value reduction |
| 7 | reduceMin | ReduceMin | Minimum value reduction |
| 8 | reduceProduct | ReduceProd | Product reduction |
| 9 | reduceSum | ReduceSum | Sum reduction |
| 10 | reduceSumSquare | ReduceSumSquare | Sum of squares reduction |
| 11 | argMax | ArgMax | |
| 12 | argMin | ArgMin | |

## RNN (2 / 4)

| # | WebNN Op | ONNX Op | Note |
|---|----------|---------|------|
| 1 | gru | GRU | Support for bidirectional, activations |
| | gruCell |||
| 2 | lstm | LSTM | Support for bidirectional, peephole, activations |
| | lstmCell |||


## Shape (4)

| # | WebNN Op | ONNX Op | Note |
|---|----------|---------|------|
| 1 | flatten (reshape emulation) | Flatten | Flattens input to 2D |
| 2 | reshape | Reshape | Direct shape manipulation |
| 3 | squeeze (reshape emulation) | Squeeze | Removes dimensions of size 1 |
| 4 | unsqueeze (reshape emulation) | Unsqueeze | Inserts dimensions of size 1 |

## Tensor (7)

| # | WebNN Op | ONNX Op | Note |
|---|----------|---------|------|
| 1 | gather | Gather | Gathers elements along an axis |
| 2 | gatherElements | GatherElements | Gathers elements based on indices tensor |
| 3 | gatherND | GatherND | Only batch_dims=0 supported |
| 4 | scatterElements | ScatterElements | Only reduction='none' supported |
| 5 | scatterND | ScatterND | Only reduction='none' supported |
| 6 | split | Split | Splits tensor into multiple parts |
| 7 | transpose | Transpose | Permutes dimensions |

## Misc (5)

| # | WebNN Op | ONNX Op | Note |
|---|----------|---------|------|
| 1 | cast | Cast | |
| 2 | concat | Concat | Concatenates tensors along specified axis |
| 3 | constant | Constant | Creates constant tensor |
| 4 | cumulativeSum | CumSum | |
| | | Dropout | Decomposed to WebNN ops |
| | | Einsum | Matmul-like einsum equations for direct mapping |
| 5 | resample2d | Resize | |
| | | Shape | Decomposed to WebNN ops |

## Others 
| # | WebNN Op | ONNX Op | Note |
|---|----------|---------|------|

| | | QLinearConv | |
| | | MaxUnpool | |
| | | GridSample | |
| | | DepthToSpace | |
| | | SpaceToDepth | |
| | |  | |
| | | CenterCropPad | |
| | | ThresholdedRelu | |
| | | OneHot | |
| | | ConstantOfShape | |

---

This comprehensive mapping covers all operations implemented in the webnn-code-generator project across all subdirectories. Some ONNX operators that don't have direct WebNN counterparts are implemented through decomposition into multiple WebNN operations or special handling.