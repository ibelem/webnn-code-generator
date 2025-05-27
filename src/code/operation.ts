import { binary_js } from './operation/binary';
import { averagePool2d_js } from './operation/averagePool2d';
import { clamp_js } from './operation/clamp';
import { conv2d_js } from './operation/conv2d';
import { convTranspose2d_js } from './operation/convTranspose2d';
import { gemm_js } from './operation/gemm';
import { reshape_js } from './operation/reshape';
import { unary_js } from './operation/unary';
import { logical_js } from './operation/logical';
import { resize_js } from './operation/resize';
import { transpose_js } from './operation/transpose';
import { softmax_js } from './operation/softmax';
import { prelu_js } from './operation/activation/prelu';
import { relu_js } from './operation/activation/relu';
import { sigmoid_js } from './operation/activation/sigmoid';
import { elu_js } from './operation/activation/elu';
import { gelu_js } from './operation/activation/gelu';
import { tanh_js } from './operation/activation/tanh';
import { hardSigmoid_js } from './operation/activation/hardSigmoid';
import { hardSwish_js } from './operation/activation/hardSwish';
import { leakyRelu_js } from './operation/activation/leakyRelu';
import { softplus_js } from './operation/activation/softplus';
import { softsign_js } from './operation/activation/softsign';

const opHandlers: Record<string, (node: any, toJsVarName: (name: string) => string) => string> = {
  // 7 element-wise binary
  Add: (node, toJsVarName) => binary_js(node, toJsVarName, 'add'),
  Sub: (node, toJsVarName) => binary_js(node, toJsVarName, 'sub'),
  Mul: (node, toJsVarName) => binary_js(node, toJsVarName, 'mul'),
  Div: (node, toJsVarName) => binary_js(node, toJsVarName, 'div'),
  Max: (node, toJsVarName) => binary_js(node, toJsVarName, 'max'),
  Min: (node, toJsVarName) => binary_js(node, toJsVarName, 'min'),
  Pow: (node, toJsVarName) => binary_js(node, toJsVarName, 'pow'),

  // 15 element-wise unary
  Abs: (node, toJsVarName) => unary_js(node, toJsVarName, 'abs'),
  Ceil: (node, toJsVarName) => unary_js(node, toJsVarName, 'ceil'),
  Cos: (node, toJsVarName) => unary_js(node, toJsVarName, 'cos'),
  Erf: (node, toJsVarName) => unary_js(node, toJsVarName, 'erf'),
  Exp: (node, toJsVarName) => unary_js(node, toJsVarName, 'exp'),
  Floor: (node, toJsVarName) => unary_js(node, toJsVarName, 'floor'),
  Identity: (node, toJsVarName) => unary_js(node, toJsVarName, 'identity'),
  Log: (node, toJsVarName) => unary_js(node, toJsVarName, 'log'),
  Neg: (node, toJsVarName) => unary_js(node, toJsVarName, 'neg'),
  Round: (node, toJsVarName) => unary_js(node, toJsVarName, 'round'),
  Reciprocal: (node, toJsVarName) => unary_js(node, toJsVarName, 'reciprocal'),
  Sin: (node, toJsVarName) => unary_js(node, toJsVarName, 'sin'),
  Sign: (node, toJsVarName) => unary_js(node, toJsVarName, 'sign'),
  Sqrt: (node, toJsVarName) => unary_js(node, toJsVarName, 'sqrt'),
  Tan: (node, toJsVarName) => unary_js(node, toJsVarName, 'tan'),

  // https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/webnn/builders/op_builder_factory.cc

  // {"ArgMax": "argMax"}, {"ArgMin": "argMin"}, {"BatchNormalization": "batchNormalization"},
  // {"Cast": "cast"}, {"Concat": "concat"}, {"CumSum": "cumulativeSum"},
  // {"DequantizeLinear": "dequantizeLinear"}, {"DynamicQuantizeLinear": "dynamicQuantizeLinear"},
  // {"Einsum": "matmul"}, {"Expand": "expand"}, {"Flatten": "reshape"},
  // {"Gather": "gather"}, {"GatherElements": "gatherElements"},
  // {"GatherND": "gatherND"}, {"GlobalMaxPool": "maxPool2d"}, {"GlobalLpPool": "l2Pool2d"}, {"GRU": "gru"},
  // {"InstanceNormalization": "instanceNormalization"}, {"LayerNormalization": "layerNormalization"},
  // {"LpPool": "l2Pool2d"}, {"LSTM": "lstm"}, {"MatMul": "matmul"}, {"Max": "max"}, {"MaxPool": "maxPool2d"},
  // {"Pad": "pad"}, {"QuantizeLinear": "quantizeLinear"}, {"Reciprocal": "reciprocal"},
  // {"ReduceL1": "reduceL1"}, {"ReduceL2": "reduceL2"}, {"ReduceLogSum": "reduceLogSum"},
  // {"ReduceLogSumExp": "reduceLogSumExp"}, {"ReduceMax": "reduceMax"},
  // {"ReduceMean": "reduceMean"}, {"ReduceMin": "reduceMin"},
  // {"ReduceProd": "reduceProduct"}, {"ReduceSum": "reduceSum"}, {"ReduceSumSquare": "reduceSumSquare"},
  // {"ScatterElements": "scatterElements"}, {"ScatterND": "scatterND"},
  // {"Shape": "slice"}, {"Slice": "slice"}, {"Split": "split"}, {"Squeeze": "reshape"},
  // {"Tile": "tile"}, {"Trilu": "triangular"}, {"Unsqueeze": "reshape"}, {"Where": "where"},

  // Activation
  Elu: elu_js,
  Gelu: gelu_js,
  HardSigmoid: hardSigmoid_js,
  HardSwish: hardSwish_js,
  LeakyRelu: leakyRelu_js,
  PRelu: prelu_js,
  Relu: relu_js,
  Sigmoid: sigmoid_js,
  Softplus: softplus_js,
  Softsign: softsign_js,
  Tanh: tanh_js,
  
  // ArgMax, ArgMin

  // Cast

  // Clip
  Clip: clamp_js,

  // Conv
  Conv: conv2d_js,
  // ConvInteger
  ConvTranspose: convTranspose2d_js,
  // // TF Lite ops
  Conv2D: conv2d_js,

  // Concat

  // CumSum

  // Dropout
  Dropout: (node, toJsVarName) => unary_js(node, toJsVarName, 'identity'),

  // DequantizeLinear, QuantizeLinear, DynamicQuantizeLinear

  // Einsum

  // Expand

  // Gather

  // GatherElements

  // GatherND

  // GroupQueryAttention

  // Flatten

  // Gemm, MatMul
  Gemm: gemm_js,
  // Matmul, MatMulInteger

  // GRU

  // 9 element-wise logical without NotEqual
  Equal: (node, toJsVarName) => logical_js(node, toJsVarName, 'equal'),
  // No NotEqual in ONNX model
  // NotEqual: (node, toJsVarName) => logical_js(node, toJsVarName, 'notEqual'),
  Greater: (node, toJsVarName) => logical_js(node, toJsVarName, 'greater'),
  GreaterOrEqual: (node, toJsVarName) => logical_js(node, toJsVarName, 'greaterOrEqual'),
  Less: (node, toJsVarName) => logical_js(node, toJsVarName, 'lesser'),
  LessOrEqual: (node, toJsVarName) => logical_js(node, toJsVarName, 'lesserOrEqual'),
  Not: (node, toJsVarName) => logical_js(node, toJsVarName, 'logicalNot'),
  And: (node, toJsVarName) => logical_js(node, toJsVarName, 'logicalAnd'),
  Or: (node, toJsVarName) => logical_js(node, toJsVarName, 'logicalOr'),
  Xor: (node, toJsVarName) => logical_js(node, toJsVarName, 'logicalXor'),

  // LRN

  // LSTM

  // MatMulNBits

  // Max, Min

  // MultiHeadAttention

  // Normalization
  // BatchNormalization, InstanceNormalization, LayerNormalization, SimplifiedLayerNormalization, SkipSimplifiedLayerNormalization
  
  // Pad

  // Pooling
  AveragePool: averagePool2d_js,
  GlobalAveragePool: averagePool2d_js,
  // GlobalMaxPool, GlobalLpPool, LpPool, MaxPool
  // // TFLite op
  AveragePool2D: averagePool2d_js,

  // Reduction
  // ReduceL1, ReduceL2, ReduceLogSum, ReduceLogSumExp, 
  // ReduceMax, ReduceMean, ReduceMin, ReduceProd, ReduceSum, ReduceSumSquare

  // Reshape
  Reshape: reshape_js,

  // Resize
  Resize: resize_js,

  // RotaryEmbedding

  // ScatterElements

  // ScatterND

  // Shape

  // Slice

  // Softmax
  Softmax: softmax_js,

  // Split

  // Squeeze, Unsqueeze

  // Tile

  // Transpose
  Transpose: transpose_js,

  // Trilu
};

export { opHandlers };