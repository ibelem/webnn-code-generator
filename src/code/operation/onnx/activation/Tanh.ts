import {
  getInputVars,
  getOutputVars
} from '../../operation-utils';

/**
 * Generate JavaScript code for a WebNN tanh operation from ONNX Tanh node info.
 * https://www.w3.org/TR/webnn/#api-mlgraphbuilder-tanh-method
 */
export function Tanh(
  node: any,
  toJsVarName: (name: string) => string,
  options?: { [key: string]: any } = {}
): string {
  const inputVars = getInputVars(node, toJsVarName);
  const outputVars = getOutputVars(node, toJsVarName);

  return `
    const ${outputVars[0]} = builder.tanh(
      ${inputVars[0]}
    );`;
}