import {
  getInputVars,
  getOutputVars,
  getAttrValue
} from '../../operation-utils';

/**
 * Generate JavaScript code for a WebNN elu operation from ONNX Elu node info.
 * https://www.w3.org/TR/webnn/#api-mlgraphbuilder-elu
 * https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/webnn/builders/impl/activation_op_builder.cc
 */

export function Elu(
  node: any,
  toJsVarName: (name: string) => string
): string {
  const inputVars = getInputVars(node, toJsVarName);
  const outputVars = getOutputVars(node, toJsVarName);

  // Use getAttrValue for robust alpha extraction (default 1.0)
  const alpha = getAttrValue(node, 'alpha', 1.0);

  // Add label for debugging if node.name exists
  const opts = `{ alpha: ${alpha}, label: '${node.name || ''}' }`;

  return `
    const ${outputVars[0]} = builder.elu(
      ${inputVars[0]},
      ${opts}
    );`;
}