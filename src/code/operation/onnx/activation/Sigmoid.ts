import {
  getInputVars,
  getOutputVars
} from '../../operation-utils';

/**
 * Generate JavaScript code for a WebNN sigmoid operation from ONNX Sigmoid node info.
 * https://www.w3.org/TR/webnn/#api-mlgraphbuilder-sigmoid-method
 * https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/webnn/builders/impl/activation_op_builder.cc
 */

export function Sigmoid(
  node: any,
  toJsVarName: (name: string) => string
): string {
  const inputVars = getInputVars(node, toJsVarName);
  const outputVars = getOutputVars(node, toJsVarName);

  // Only add label if node.name exists
  const opts = node.name ? `{ label: '${node.name}' }` : '{}';

  return `
    const ${outputVars[0]} = builder.sigmoid(
      ${inputVars[0]},
      ${opts}
    );
`;
}