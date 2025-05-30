import {
  getInputVars,
  getOutputVars
} from '../../operation-utils';

/**
 * Generate JavaScript code for a WebNN sigmoid operation from ONNX Sigmoid node info.
 * https://www.w3.org/TR/webnn/#api-mlgraphbuilder-sigmoid-method
 */
export function Sigmoid(
  node: any,
  toJsVarName: (name: string) => string
): string {
  const inputVars = getInputVars(node, toJsVarName);
  const outputVars = getOutputVars(node, toJsVarName);

  return `
    const ${outputVars[0]} = builder.sigmoid(
      ${inputVars[0]}
    );`;
}