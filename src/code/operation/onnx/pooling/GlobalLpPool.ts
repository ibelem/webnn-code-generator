import {
  getInputVars,
  getOutputVars,
  getShape
} from '../../operation-utils';

/**
 * Generate JavaScript code for a WebNN global l2Pool2d operation from ONNX GlobalLpPool node info.
 * https://www.w3.org/TR/webnn/#api-mlgraphbuilder-pool2d-l2
 * Only supports p=2 (L2 norm), as required by WebNN.
 */
export function GlobalLpPool(
  node: any,
  toJsVarName: (name: string) => string,
  options: { nhwc?: boolean } = {}
): string {
  const inputVars = getInputVars(node, toJsVarName);
  const outputVars = getOutputVars(node, toJsVarName);
  const inputShape = getShape(node, 0);

  // Only support p=2 (L2 norm)
  let p = 2;
  for (const attr of node.attributes || []) {
    if (attr.name === 'p') {
      p = typeof attr.value === 'number' ? attr.value : Number(attr.value?.value);
      break;
    }
  }
  if (p !== 2) {
    return `// Only L2 pooling (p=2) is supported by WebNN.`;
  }

  const nhwc = !!options.nhwc;
  const layout = nhwc ? "'nhwc'" : "'nchw'";
  // For global pooling, windowDimensions is the spatial dims of the input
  // NCHW: [b, c, h, w] -> [h, w], NHWC: [b, h, w, c] -> [h, w]
  const windowDims = nhwc
    ? [inputShape[1], inputShape[2]]
    : [inputShape[2], inputShape[3]];

  return `
    const ${outputVars[0]} = builder.l2Pool2d(
      ${inputVars[0]},
      {
        windowDimensions: [${windowDims.join(', ')}],
        layout: ${layout}
      }
    );`;
}