import {
  getInputVars,
  getOutputVars,
  getAttr
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
  toJsVarName: (name: string) => string
): string {
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
  let axes = getAttr(node, 'axes', undefined);
  // Try to get axes from input[1] if not present in attributes
  if (!axes && node.inputs?.[1]?.initializer) {
    axes = node.inputs[1].initializer;
  }
  // keepdims: ONNX default is 1 (true)
  const keepDims = !!getAttr(node, 'keepdims', 1);

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
    );
`;
  }
  if (builderMethod === 'reduceLogSumExp') {
    return `
    const ${outputVars[0]} = builder.log(
      builder.reduceSum(builder.exp(${inputVars[0]}), { ${opts.join(', ')} })
    );
`;
  }

  // Normal reduction op
  return `
    const ${outputVars[0]} = builder.${builderMethod}(
      ${inputVars[0]},
      { ${opts.join(', ')} }
    );
`;
}