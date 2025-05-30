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
  toJsVarName: (name: string) => string
): string {
  const inputVars = getInputVars(node, toJsVarName);
  const outputVars = getOutputVars(node, toJsVarName);

  // Add label for debugging if node.name exists
  const opts = `{ label: '${node.name || ''}' }`;

  return `
    const ${outputVars[0]} = builder.tanh(
      ${inputVars[0]},
      ${opts}
    );
  `;
}