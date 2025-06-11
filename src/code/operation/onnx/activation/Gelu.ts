import {
  getInputVars,
  getOutputVars
} from '../../operation-utils';

/**
 * Generate JavaScript code for a WebNN gelu operation from ONNX Gelu node info.
 * https://www.w3.org/TR/webnn/#api-mlgraphbuilder-gelu-method
 * https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/webnn/builders/impl/activation_op_builder.cc
 */

export function Gelu(
  node: any,
  toJsVarName: (name: string) => string
): string {
  const inputVars = getInputVars(node, toJsVarName);
  const outputVars = getOutputVars(node, toJsVarName);

  // Only add label for debugging if node.name exists
  const opts = node.name ? `{ label: '${node.name}' }` : '{}';

  return `
    const ${outputVars[0]} = builder.gelu(
      ${inputVars[0]},
      ${opts}
    );
`;
}