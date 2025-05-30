import {
  getInputVars,
  getOutputVars
} from '../../operation-utils';

/**
 * Generate JavaScript code for a WebNN elu operation from ONNX Elu node info.
 * https://www.w3.org/TR/webnn/#api-mlgraphbuilder-elu
 */
export function Elu(
  node: any,
  toJsVarName: (name: string) => string,
  _options: { [key: string]: any } = {}
): string {
  const inputVars = getInputVars(node, toJsVarName);
  const outputVars = getOutputVars(node, toJsVarName);

  // Default alpha is 1.0 for ONNX Elu
  let alpha = 1.0;
  for (const attr of node.attributes || []) {
    if (attr.name === 'alpha') {
      alpha = Number(attr.f ?? 1.0);
      break;
    }
  }

  return `
    const ${outputVars[0]} = builder.elu(
      ${inputVars[0]},
      { alpha: ${alpha} }
    );`;
}