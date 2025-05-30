import {
  getInputVars,
  getOutputVars
} from '../../operation-utils';

/**
 * Generate JavaScript code for a WebNN softsign operation from ONNX Softsign node info.
 * https://www.w3.org/TR/webnn/#api-mlgraphbuilder-softsign-method
 */
export function Softsign(
  node: any,
  toJsVarName: (name: string) => string
): string {
  const inputVars = getInputVars(node, toJsVarName);
  const outputVars = getOutputVars(node, toJsVarName);

  // Add label for debugging if node.name exists
  const opts = `{ label: '${node.name || ''}' }`;

  return `
    const ${outputVars[0]} = builder.softsign(
      ${inputVars[0]},
      ${opts}
    );
  `;
}