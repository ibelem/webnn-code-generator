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

  return `
    const ${outputVars[0]} = builder.softsign(
      ${inputVars[0]}
    );`;
}