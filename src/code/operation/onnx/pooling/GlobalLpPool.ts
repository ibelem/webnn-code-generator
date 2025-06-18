import {
  getInputVars,
  getOutputVars,
  getShape,
  getAttrValue
} from '../../operation-utils';

/**
 * Generate JavaScript code for a WebNN global l2Pool2d operation from ONNX GlobalLpPool node info.
 * https://www.w3.org/TR/webnn/#api-mlgraphbuilder-pool2d-l2
 * https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/webnn/builders/impl/pool_op_builder.cc
 * Only supports p=2 (L2 norm), as required by WebNN.
 */
export function GlobalLpPool(
  node: any,
  toJsVarName: (name: string) => string,
  options: { nhwc?: boolean } = {}
): string {
  const inputVars = getInputVars(node, toJsVarName);
  const outputVars = getOutputVars(node, toJsVarName);
  const nhwc = !!options.nhwc;

  // Only support p=2 (L2 norm)
  let p = getAttrValue(node, 'p', 2);

  if (p !== 2) {
    return `// Only L2 pooling (p=2) is supported by WebNN.`;
  }

  // Get input shape and compute windowDimensions for global pooling
  const { shape: inputShape } = getShape(node, 0, nhwc);
  // NCHW: [b, c, h, w] -> [h, w], NHWC: [b, h, w, c] -> [h, w]
  const windowDims = nhwc
    ? [inputShape[1], inputShape[2]]
    : [inputShape[2], inputShape[3]];

  const opts: string[] = [
    `windowDimensions: [${windowDims.join(', ')}]`,
    `layout: '${nhwc ? 'nhwc' : 'nchw'}'`
  ];
  if (node.name) {
    opts.push(`label: '${node.name}'`);
  }

  return `
    const ${outputVars[0]} = builder.l2Pool2d(
      ${inputVars[0]},
      {
        ${opts.join(',\n    ')}
      }
    );`;
}