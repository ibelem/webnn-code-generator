import {
  getInputVars,
  getOutputVars,
  getAttrValue
} from '../../operation-utils';

/**
 * Generate JavaScript code for a WebNN leakyRelu operation from ONNX LeakyRelu node info.
 * https://www.w3.org/TR/webnn/#api-mlgraphbuilder-leakyrelu
 * https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/webnn/builders/impl/activation_op_builder.cc
 */

export function LeakyRelu(
  node: any,
  toJsVarName: (name: string) => string
): string {
  const inputVars = getInputVars(node, toJsVarName);
  const outputVars = getOutputVars(node, toJsVarName);

  // Default alpha is 0.01 for ONNX LeakyRelu
  // ONNX default alpha is 0.01
  const alpha = getAttrValue(node, 'alpha', 0.01);

  // Add label for debugging if node.name exists
  const opts = node.name
    ? `{ alpha: ${alpha}, label: '${node.name}' }`
    : `{ alpha: ${alpha} }`;

  return `
    const ${outputVars[0]} = builder.leakyRelu(
      ${inputVars[0]},
      ${opts}
    );
`;
}