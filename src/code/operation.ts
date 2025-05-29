import { add, sub, mul, div, max, min, pow } from './operation/binary';
import { argMax, argMin } from './operation/argMinMax';
import { averagePool2d } from './operation/averagePool2d';
import { clamp } from './operation/clamp';
import { conv2d } from './operation/conv2d';
import { convTranspose2d } from './operation/convTranspose2d';
import { gemm } from './operation/gemm';
import { reshape } from './operation/reshape';
import { resize } from './operation/resize';
import { transpose } from './operation/transpose';
import { softmax } from './operation/softmax';
import { prelu } from './operation/activation/prelu';
import { relu } from './operation/activation/relu';
import { sigmoid } from './operation/activation/sigmoid';
import { elu } from './operation/activation/elu';
import { gelu } from './operation/activation/gelu';
import { tanh } from './operation/activation/tanh';
import { hardSigmoid } from './operation/activation/hardSigmoid';
import { hardSwish } from './operation/activation/hardSwish';
import { leakyRelu } from './operation/activation/leakyRelu';
import {
  equal, greater, greaterOrEqual, less, lessOrEqual,
  not, and, or, xor
} from './operation/logical';
import { softplus } from './operation/activation/softplus';
import { softsign } from './operation/activation/softsign';
import {
  abs, ceil, cos, erf, exp, floor, identity, log,
  neg, round, reciprocal, sin, sign, sqrt, tan
} from './operation/unary';
import { dequantizeLinear } from './operation/quantization/dequantizeLinear';
import { quantizeLinear } from './operation/quantization/quantizeLinear';
import { matmul } from './operation/matmul';

const opHandlers: Record<string, (node: any, toJsVarName: (name: string) => string) => string> = {
  // 7 element-wise binary
  Add: add,
  Sub: sub,
  Mul: mul,
  Div: div,
  Max: max,
  Min: min,
  Pow: pow,

  // 15 element-wise unary
  Abs: abs,
  Ceil: ceil,
  Cos: cos,
  Erf: erf,
  Exp: exp,
  Floor: floor,
  Identity: identity,
  Log: log,
  Neg: neg,
  Round: round,
  Reciprocal: reciprocal,
  Sin: sin,
  Sign: sign,
  Sqrt: sqrt,
  Tan: tan,

  // https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/webnn/builders/op_builder_factory.cc

  // {"BatchNormalization": "batchNormalization"},
  // {"Cast": "cast"}, {"Concat": "concat"}, {"CumSum": "cumulativeSum"},
  // {"DynamicQuantizeLinear": "dynamicQuantizeLinear"},
  // {"Einsum": "matmul"}, {"Expand": "expand"}, {"Flatten": "reshape"},
  // {"Gather": "gather"}, {"GatherElements": "gatherElements"},
  // {"GatherND": "gatherND"}, {"GlobalMaxPool": "maxPool2d"}, {"GlobalLpPool": "l2Pool2d"}, {"GRU": "gru"},
  // {"InstanceNormalization": "instanceNormalization"}, {"LayerNormalization": "layerNormalization"},
  // {"LpPool": "l2Pool2d"}, {"LSTM": "lstm"}, {"MaxPool": "maxPool2d"},
  // {"Pad": "pad"}, {"Reciprocal": "reciprocal"},
  // {"ReduceL1": "reduceL1"}, {"ReduceL2": "reduceL2"}, {"ReduceLogSum": "reduceLogSum"},
  // {"ReduceLogSumExp": "reduceLogSumExp"}, {"ReduceMax": "reduceMax"},
  // {"ReduceMean": "reduceMean"}, {"ReduceMin": "reduceMin"},
  // {"ReduceProd": "reduceProduct"}, {"ReduceSum": "reduceSum"}, {"ReduceSumSquare": "reduceSumSquare"},
  // {"ScatterElements": "scatterElements"}, {"ScatterND": "scatterND"},
  // {"Shape": "slice"}, {"Slice": "slice"}, {"Split": "split"}, {"Squeeze": "reshape"},
  // {"Tile": "tile"}, {"Trilu": "triangular"}, {"Unsqueeze": "reshape"}, {"Where": "where"},

  // Activation
  Elu: elu,
  Gelu: gelu,
  HardSigmoid: hardSigmoid,
  HardSwish: hardSwish,
  LeakyRelu: leakyRelu,
  PRelu: prelu,
  Relu: relu,
  Sigmoid: sigmoid,
  Softplus: softplus,
  Softsign: softsign,
  Tanh: tanh,
  
  // ArgMax, ArgMin
  ArgMax: argMax,
  ArgMin: argMin,

  // Cast

  // Clip
  Clip: clamp,

  // Conv
  Conv: conv2d,
  // ConvInteger
  ConvTranspose: convTranspose2d,
  // // TF Lite ops
  Conv2D: conv2d,

  // Concat

  // CumSum

  // Dropout
  Dropout: identity,

  // DynamicQuantizeLinear
  DequantizeLinear: dequantizeLinear,
  QuantizeLinear: quantizeLinear,

  // Einsum

  // Expand

  // Gather

  // GatherElements

  // GatherND

  // GroupQueryAttention

  // Flatten

  // MatMulInteger
  Gemm: gemm,
  MatMul: matmul,

  // GRU

  // 9 element-wise logical without NotEqual
  Equal: equal,
  // No NotEqual in ONNX model
  // NotEqual: (node, toJsVarName) => logical(node, toJsVarName, 'notEqual'),
  Greater: greater,
  GreaterOrEqual: greaterOrEqual,
  Less: less,
  LessOrEqual: lessOrEqual,
  Not: not,
  And: and,
  Or: or,
  Xor: xor,

  // LRN

  // LSTM

  // MatMulNBits

  // MultiHeadAttention

  // Normalization
  // BatchNormalization, InstanceNormalization, LayerNormalization, SimplifiedLayerNormalization, SkipSimplifiedLayerNormalization
  
  // Pad

  // Pooling
  AveragePool: averagePool2d,
  GlobalAveragePool: averagePool2d,
  // GlobalMaxPool, GlobalLpPool, LpPool, MaxPool
  // // TFLite op
  AveragePool2D: averagePool2d,

  // Reduction
  // ReduceL1, ReduceL2, ReduceLogSum, ReduceLogSumExp, 
  // ReduceMax, ReduceMean, ReduceMin, ReduceProd, ReduceSum, ReduceSumSquare

  // Reshape
  Reshape: reshape,

  // Resize
  Resize: resize,

  // RotaryEmbedding

  // ScatterElements

  // ScatterND

  // Shape

  // Slice

  // Softmax
  Softmax: softmax,

  // Split

  // Squeeze, Unsqueeze

  // Tile

  // Transpose
  Transpose: transpose,

  // Trilu
};

  // Some ONNX ops are supported by decomposed WebNN ops
  // {"ConvInteger", {"cast", "conv2d", "dequantizeLinear"}},
  // {"GroupQueryAttention", {"add", "cast", "concat", "constant", "cumulativeSum", 
  //    "div", "expand", "lesser", "matmul", "reshape", "scatterND", "softmax", "transpose", "where"}},
  // {"LRN", {"add", "averagePool2d", "div", "mul", "pad", "pow", "transpose"}},
  // {"MatMulInteger", {"cast", "dequantizeLinear", "matmul"}},
  // {"MatMulNBits", {"add", "dequantizeLinear", "matmul", "reshape", "transpose"}},
  // {"MultiHeadAttention", {"add", "cast", "concat", "constant", "div", "matmul", "reshape", "softmax", "transpose"}},
  // {"RotaryEmbedding", {"add", "concat", "gather", "mul", "reshape", "slice", "split"}},
  // {"SimplifiedLayerNormalization", {"add", "div", "mul", "pow", "reduceMean", "sqrt"}},
  // {"SkipSimplifiedLayerNormalization", {"add", "div", "mul", "pow", "reduceMean", "sqrt"}},

export { opHandlers };