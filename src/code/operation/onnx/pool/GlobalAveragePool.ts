import {
  getInputVars,
  getOutputVars
} from '../../operation-utils';

/**
 * Generate JavaScript code for a WebNN averagePool2d operation from ONNX GlobalAveragePool node info.
 * https://www.w3.org/TR/webnn/#api-mlgraphbuilder-pool2d-average
 */
export function GlobalAveragePool(
  node: any,
  toJsVarName: (name: string) => string,
  options?: { [key: string]: any } = {}
): string {
  const inputVars = getInputVars(node, toJsVarName);
  const outputVars = getOutputVars(node, toJsVarName);

  // For GlobalAveragePool, just call averagePool2d with no options (global pooling)
  return `
    const ${outputVars[0]} = builder.averagePool2d(
      ${inputVars[0]}
    );`;
}