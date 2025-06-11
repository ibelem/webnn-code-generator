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

## Activation

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

## Attention

| # | WebNN Op | ONNX Op | Note |
|---|----------|---------|------|
| 1 | N/A | GroupQueryAttention | Decomposed to WebNN subgraph |
| 2 | N/A | MultiHeadAttention | Decomposed to WebNN subgraph |
| 3 | N/A | RotaryEmbedding | Decomposed to WebNN subgraph |
| 4 | N/A | ScaledDotProductAttention | Helper function for attention ops |

## Convolution

| # | WebNN Op | ONNX Op | Note |
|---|----------|---------|------|
| 1 | conv2d | Conv | Support for NCHW/NHWC layouts |
| 2 | conv2d | DepthwiseConv2d | Implemented using groups parameter |
| 3 | N/A | ConvTranspose | Not directly supported, requires decomposition |

## Elementwise Binary

| # | WebNN Op | ONNX Op | Note |
|---|----------|---------|------|
| 1 | add | Add | Addition of two tensors |
| 2 | div | Div | Division of two tensors |
| 3 | max | Max | Element-wise maximum of two tensors |
| 4 | min | Min | Element-wise minimum of two tensors |
| 5 | mul | Mul | Multiplication of two tensors |
| 6 | pow | Pow | Element-wise power operation |
| 7 | sub | Sub | Subtraction of two tensors |

## Elementwise Logical

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

## Elementwise Unary

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
| 12 | round | Round | Rounds to nearest integer |
| 13 | sign | Sign | Returns sign of the input |
| 14 | sin | Sin | Sine function |
| 15 | sqrt | Sqrt | Square root function |
| 16 | tan | Tan | Tangent function |

## Elementwise Others

| # | WebNN Op | ONNX Op | Note |
|---|----------|---------|------|
| 3 | clamp | Clip | Clamps values between min and max |
| 1 | where | Where | Ternary operation (condition ? x : y) |

## Linear

| # | WebNN Op | ONNX Op | Note |
|---|----------|---------|------|
| 1 | clamp | Clamp | Clamps values between min and max |
| 2 | gemm | Gemm | General matrix multiplication with bias |
| 3 | matmul | MatMul | Matrix multiplication |
| 4 | N/A | MatMulInteger | Quantized matmul, decomposed |
| 5 | N/A | MatMulNBits | Decomposed to dequantizeLinear + matmul |
| 6 | triangular | Triangular | Upper or lower triangular matrix |

## Normalization

| # | WebNN Op | ONNX Op | Note |
|---|----------|---------|------|
| 1 | batchNormalization | BatchNormalization | `axis` option set based on layout |
| 2 | instanceNormalization | InstanceNormalization | `layout` and `epsilon` options supported |
| 3 | layerNormalization | LayerNormalization | Supports axes and epsilon options |
| 4 | N/A | LRN | Decomposed to multiple WebNN ops |
| 5 | N/A | MeanVarianceNormalization | Decomposed to WebNN ops |
| 6 | N/A | SimplifiedLayerNormalization | Decomposed to WebNN ops |
| 7 | N/A | SkipLayerNormalization | Decomposed to WebNN ops |
| 8 | N/A | SkipSimplifiedLayerNormalization | Decomposed to WebNN ops |

## Pooling

| # | WebNN Op | ONNX Op | Note |
|---|----------|---------|------|
| 1 | averagePool2d | AveragePool | 2D only, layout sensitive |
| 2 | averagePool2d | GlobalAveragePool | Uses input shape as window dimensions |
| 3 | l2Pool2d | GlobalLpPool | Only p=2 supported |
| 4 | l2Pool2d | LpPool | Only p=2 supported |
| 5 | maxPool2d | GlobalMaxPool | Uses input shape as window dimensions |
| 6 | maxPool2d | MaxPool | 2D only, layout sensitive |

## Quantization

| # | WebNN Op | ONNX Op | Note |
|---|----------|---------|------|
| 1 | cast | Cast | Data type conversion |
| 2 | dequantizeLinear | DequantizeLinear | Converts quantized to float |
| 3 | quantizeLinear | QuantizeLinear | Converts float to quantized |

## Reduction

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

## RNN

| # | WebNN Op | ONNX Op | Note |
|---|----------|---------|------|
| 1 | gru | GRU | Support for bidirectional, activations |
| 2 | lstm | LSTM | Support for bidirectional, peephole, activations |

## Shape

| # | WebNN Op | ONNX Op | Note |
|---|----------|---------|------|
| 1 | expand | Expand | Broadcasts tensor to new shape |
| 2 | reshape | Flatten | Flattens input to 2D |
| 3 | reshape | Reshape | Direct shape manipulation |
| 4 | reshape | Squeeze | Removes dimensions of size 1 |
| 5 | reshape | Unsqueeze | Inserts dimensions of size 1 |

## Tensor

| # | WebNN Op | ONNX Op | Note |
|---|----------|---------|------|
| 1 | concat | Concat | Concatenates tensors along specified axis |
| 2 | constantOfShape | ConstantOfShape | Creates tensor of given shape with constant values |
| 3 | expand | Expand | Broadcasts tensor to new shape |
| 4 | gather | Gather | Gathers elements along an axis |
| 5 | gatherElements | GatherElements | Gathers elements based on indices tensor |
| 6 | gatherND | GatherND | Only batch_dims=0 supported |
| 7 | nonMaxSuppression | NonMaxSuppression | Filters bounding boxes based on scores |
| 8 | pad | Pad | Supports constant, reflection, and edge modes |
| 9 | scatterElements | ScatterElements | Only reduction='none' supported |
| 10 | scatterND | ScatterND | Only reduction='none' supported |
| 11 | slice | Slice | Extracts slice with optional steps |
| 12 | split | Split | Splits tensor into multiple parts |
| 13 | tile | Tile | Repeats tensor along dimensions |
| 14 | transpose | Transpose | Permutes dimensions |

## Misc

| # | WebNN Op | ONNX Op | Note |
|---|----------|---------|------|
| 1 | constant | Constant | Creates constant tensor |
| 2 | N/A | CenterCropPad | Not directly supported |
| 3 | N/A | DepthToSpace | Decomposed to WebNN ops |
| 4 | N/A | OneHot | Not directly supported |
| 5 | N/A | Range | Decomposed to WebNN ops |
| 6 | N/A | Resize | Decomposed to WebNN ops for various resize modes |
| 7 | N/A | SpaceToDepth | Not directly supported |
| 8 | N/A | ThresholdedRelu | Emulated with where, constant |

---

This comprehensive mapping covers all operations implemented in the webnn-code-generator project across all subdirectories. Some ONNX operators that don't have direct WebNN counterparts are implemented through decomposition into multiple WebNN operations or special handling.