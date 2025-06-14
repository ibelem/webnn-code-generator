import {
  getInputVars,
  getOutputVars
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
  let alpha = 0.01;
  for (const attr of node.attributes || []) {
    if (attr.name === 'alpha') {
      // Todo
      alpha = typeof attr.f === 'number'
        ? attr.f
        : (typeof attr.value === 'number' ? attr.value : Number(attr.value?.value ?? 0.01));
      break;
    }
  }

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