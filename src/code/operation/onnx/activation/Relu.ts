import {
  getInputVars,
  getOutputVars
} from '../../operation-utils';

/**
 * Generate JavaScript code for a WebNN relu operation from ONNX Relu node info.
 * https://www.w3.org/TR/webnn/#api-mlgraphbuilder-relu-method
 */
export function Relu(
  node: any,
  toJsVarName: (name: string) => string,
  options?: { [key: string]: any } = {}
): string {
  const inputVars = getInputVars(node, toJsVarName);
  const outputVars = getOutputVars(node, toJsVarName);

  return `
    const ${outputVars[0]} = builder.relu(
      ${inputVars[0]}
    );`;
}