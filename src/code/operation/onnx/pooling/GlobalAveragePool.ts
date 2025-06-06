import {
  getInputVars,
  getOutputVars,
  getShape
} from '../../operation-utils';

/**
 * Generate JavaScript code for a WebNN averagePool2d operation from ONNX GlobalAveragePool node info.
 * https://www.w3.org/TR/webnn/#api-mlgraphbuilder-pool2d-average
 */
export function GlobalAveragePool(
  node: any,
  toJsVarName: (name: string) => string,
  options: { nhwc?: boolean } = {}
): string {
  const inputVars = getInputVars(node, toJsVarName);
  const outputVars = getOutputVars(node, toJsVarName);
  const nhwc = !!options.nhwc;

  // const inputShape = getShape(node, 0, nhwc);  // Add nhwc parameter

  // For GlobalAveragePool, just call averagePool2d with no options (global pooling)
  // Add layout: 'nhwc' if needed
  return nhwc
    ? `
    const ${outputVars[0]} = builder.averagePool2d(
      ${inputVars[0]},
      { layout: 'nhwc' }
    );`
    : `
    const ${outputVars[0]} = builder.averagePool2d(
      ${inputVars[0]}
    );`;
}