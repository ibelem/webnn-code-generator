import {
  getInputVars,
  getOutputVars
} from '../../operation-utils';

/**
 * Generate JavaScript code for a WebNN leakyRelu operation from ONNX LeakyRelu node info.
 * https://www.w3.org/TR/webnn/#api-mlgraphbuilder-leakyrelu
 */
export function LeakyRelu(
  node: any,
  toJsVarName: (name: string) => string,
  options?: { [key: string]: any } = {}
): string {
  const inputVars = getInputVars(node, toJsVarName);
  const outputVars = getOutputVars(node, toJsVarName);

  // Default alpha is 0.01 for ONNX LeakyRelu
  let alpha = 0.01;
  for (const attr of node.attributes || []) {
    if (attr.name === 'alpha') {
      // Todo: Handle attr.f attribute correctly from json file
      alpha = Number(attr.f ?? 0.01);
      break;
    }
  }

  return `
    const ${outputVars[0]} = builder.leakyRelu(
      ${inputVars[0]},
      { alpha: ${alpha} }
    );`;
}