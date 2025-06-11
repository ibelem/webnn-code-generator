import {
  getInputVars,
  getOutputVars
} from '../../operation-utils';

/**
 * Generate JavaScript code for WebNN where operation from ONNX Where node.
 * https://www.w3.org/TR/webnn/#api-mlgraphbuilder-where
 * https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/webnn/builders/impl/ternary_op_builder.cc
 */
export function Ternary(
  node: any,
  toJsVarName: (name: string) => string
): string {
  const inputVars = getInputVars(node, toJsVarName); // [condition, trueValue, falseValue]
  const outputVars = getOutputVars(node, toJsVarName);

  // Add label for debugging if available
  const labelOpt = node.name ? `{ label: '${node.name}' }` : '{}';

  return `
    const ${outputVars[0]} = builder.where(
      ${inputVars[0]},
      ${inputVars[1]},
      ${inputVars[2]},
      ${labelOpt}
    );
`;
}