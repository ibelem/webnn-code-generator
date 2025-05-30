import {
  getInputVars,
  getOutputVars
} from '../../operation-utils';

/**
 * Generate JavaScript code for a WebNN gelu operation from ONNX Gelu node info.
 * https://www.w3.org/TR/webnn/#api-mlgraphbuilder-gelu-method
 */
export function Gelu(
  node: any,
  toJsVarName: (name: string) => string,
  _options: { [key: string]: any } = {}
): string {
  const inputVars = getInputVars(node, toJsVarName);
  const outputVars = getOutputVars(node, toJsVarName);

  // Default mode is 'exact' for ONNX Gelu
  let mode = 'exact';
  for (const attr of node.attributes || []) {
    if (attr.name === 'approximate') {
      // Todo: Handle attr.s attribute correctly from json file
      mode = attr.s === 'tanh' ? 'tanh' : 'exact';
      break;
    }
  }

  return `
    const ${outputVars[0]} = builder.gelu(
      ${inputVars[0]},
      { mode: '${mode}' }
    );`;
}