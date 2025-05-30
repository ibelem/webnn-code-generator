import {
  getInputVars,
  getOutputVars
} from '../../operation-utils';

/**
 * Generate JavaScript code for a WebNN softplus operation from ONNX Softplus node info.
 * https://www.w3.org/TR/webnn/#api-mlgraphbuilder-softplus-method
 */
export function Softplus(
  node: any,
  toJsVarName: (name: string) => string
): string {
  const inputVars = getInputVars(node, toJsVarName);
  const outputVars = getOutputVars(node, toJsVarName);

  // Add label for debugging if node.name exists
  const opts = `{ label: '${node.name || ''}' }`;

  return `
    const ${outputVars[0]} = builder.softplus(
      ${inputVars[0]},
      ${opts}
    );
  `;
}