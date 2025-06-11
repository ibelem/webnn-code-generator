import {
  getInputVars,
  getOutputVars
} from '../../operation-utils';

/**
 * Generate JavaScript code for a WebNN prelu operation from ONNX PRelu node info.
 * https://www.w3.org/TR/webnn/#api-mlgraphbuilder-prelu
 * https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/webnn/builders/impl/activation_op_builder.cc
 */

export function PRelu(
  node: any,
  toJsVarName: (name: string) => string
): string {
  const inputVars = getInputVars(node, toJsVarName);
  const outputVars = getOutputVars(node, toJsVarName);

  // Add label for debugging if node.name exists
  const opts = node.name ? `{ label: '${node.name}' }` : '{}';

  return `
    const ${outputVars[0]} = builder.prelu(
      ${inputVars[0]},
      ${inputVars[1]},
      ${opts}
    );
`;
}