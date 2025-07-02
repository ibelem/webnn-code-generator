import {
  getInputVars,
  getOutputVars,
  getAttrValue
} from '../../operation-utils';

/**
 * Generate JavaScript code for WebNN reduction operations from ONNX reduction node info.
 * Supports: reduceL1, reduceL2, reduceLogSum, reduceLogSumExp, reduceMax, reduceMean,
 * reduceMin, reduceProduct, reduceSum, reduceSumSquare.
 * https://www.w3.org/TR/webnn/#api-mlgraphbuilder-reduce
 * https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/webnn/builders/impl/reduction_op_builder.cc
 */

export function Reduction(
  node: any,
  toJsVarName: (name: string) => string,
  options: { [key: string]: any } = {}
): string {
    const nhwc = !!options.nhwc;
  // ONNX: input, (optional) axes, (optional) keepdims, (optional) noop_with_empty_axes
  // WebNN: reduceXXX(input, {axes, keepDimensions})
  const inputVars = getInputVars(node, toJsVarName); // [input, axes?]
  const outputVars = getOutputVars(node, toJsVarName);

  // Map ONNX op type to WebNN builder method
  const opMap: Record<string, string> = {
    ReduceL1: 'reduceL1',
    ReduceL2: 'reduceL2',
    ReduceLogSum: 'reduceLogSum',
    ReduceLogSumExp: 'reduceLogSumExp',
    ReduceMax: 'reduceMax',
    ReduceMean: 'reduceMean',
    ReduceMin: 'reduceMin',
    ReduceProd: 'reduceProduct',
    ReduceSum: 'reduceSum',
    ReduceSumSquare: 'reduceSumSquare'
  };

  const opType = node.opType || node.type || node.kind;
  const builderMethod = opMap[opType];
  if (!builderMethod) {
    throw new Error(`Unsupported reduction op: ${opType}`);
  }

  // Axes: from attribute or initializer
  let axes = getAttrValue(node, 'axes', undefined);
  // Try to get axes from input[1] if not present in attributes
  if (!axes && node.inputs?.[1]?.initializer) {
    axes = node.inputs[1].initializer;
  }

  if (nhwc && Array.isArray(axes)) {
    // NCHW: [N, C, H, W] → NHWC: [N, H, W, C]
    axes = axes.map(axis => {
      if (axis === 0) return 0;      // N stays at 0
      if (axis === 1) return 3;      // C moves to 3
      if (axis === 2) return 1;      // H moves to 1
      if (axis === 3) return 2;      // W moves to 2
      // For higher rank, generalize if needed
      return axis;
    });
  }

  // keepdims: ONNX default is 1 (true)
  const keepDims = !!getAttrValue(node, 'keepdims', 1);

  // Compose options
  let opts: string[] = [`keepDimensions: ${keepDims}`];
  if (axes && axes.length > 0) {
    opts.push(`axes: [${axes.join(', ')}]`);
  }
  opts.push(`label: '${node.name}'`);

  // Special handling for ReduceLogSum and ReduceLogSumExp (decompose if needed)
  if (builderMethod === 'reduceLogSum') {
    return `
    const ${outputVars[0]} = builder.log(
      builder.reduceSum(${inputVars[0]}, { ${opts.join(', ')} })
    );`;
  }
  if (builderMethod === 'reduceLogSumExp') {
    return `
    const ${outputVars[0]} = builder.log(
      builder.reduceSum(builder.exp(${inputVars[0]}), { ${opts.join(', ')} })
    );`;
  }

  // Normal reduction op
  return `
    const ${outputVars[0]} = builder.${builderMethod}(
      ${inputVars[0]},
      { ${opts.join(', ')} }
    );`;
}

export function ReduceL1(node: any, toJsVarName: (name: string) => string, options?: any) {
  return Reduction({ ...node, opType: 'ReduceL1' }, toJsVarName, options);
}
export function ReduceL2(node: any, toJsVarName: (name: string) => string, options?: any) {
  return Reduction({ ...node, opType: 'ReduceL2' }, toJsVarName, options);
}
export function ReduceLogSum(node: any, toJsVarName: (name: string) => string, options?: any) {
  return Reduction({ ...node, opType: 'ReduceLogSum' }, toJsVarName, options);
}
export function ReduceLogSumExp(node: any, toJsVarName: (name: string) => string, options?: any) {
  return Reduction({ ...node, opType: 'ReduceLogSumExp' }, toJsVarName, options);
}
export function ReduceMax(node: any, toJsVarName: (name: string) => string, options?: any) {
  return Reduction({ ...node, opType: 'ReduceMax' }, toJsVarName, options);
}
export function ReduceMean(node: any, toJsVarName: (name: string) => string, options?: any) {
  return Reduction({ ...node, opType: 'ReduceMean' }, toJsVarName, options);
}
export function ReduceMin(node: any, toJsVarName: (name: string) => string, options?: any) {
  return Reduction({ ...node, opType: 'ReduceMin' }, toJsVarName, options);
}
export function ReduceProd(node: any, toJsVarName: (name: string) => string, options?: any) {
  return Reduction({ ...node, opType: 'ReduceProd' }, toJsVarName, options);
}
export function ReduceSum(node: any, toJsVarName: (name: string) => string, options?: any) {
  return Reduction({ ...node, opType: 'ReduceSum' }, toJsVarName, options);
}
export function ReduceSumSquare(node: any, toJsVarName: (name: string) => string, options?: any) {
  return Reduction({ ...node, opType: 'ReduceSumSquare' }, toJsVarName, options);
}