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
  toJsVarName: (name: string) => string
): string {
  const inputVars = getInputVars(node, toJsVarName);
  const outputVars = getOutputVars(node, toJsVarName);

  // Default mode is 'exact' for ONNX Gelu
  let mode = 'exact';
  for (const attr of node.attributes || []) {
    if (attr.name === 'approximate') {
      // ONNX uses 'tanh' for approximate, otherwise 'exact'
      // Todo: Handle attr.s and attr.value correctly from json file
      mode =
        (typeof attr.s === 'string' && attr.s === 'tanh') ||
        (typeof attr.value === 'string' && attr.value === 'tanh') ||
        (typeof attr.value?.value === 'string' && attr.value.value === 'tanh')
          ? 'tanh'
          : 'exact';
      break;
    }
  }

  // Add label for debugging if node.name exists
  const opts = `{ mode: '${mode}', label: '${node.name || ''}' }`;

  return `
    const ${outputVars[0]} = builder.gelu(
      ${inputVars[0]},
      ${opts}
    );
  `;
}