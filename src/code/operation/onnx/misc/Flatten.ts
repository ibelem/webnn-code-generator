import {
  getInputVars,
  getOutputVars,
  getShape
} from '../../operation-utils';

/**
 * Generate JavaScript code for WebNN from ONNX Flatten node.
 * Uses reshape to implement flatten.
 * https://www.w3.org/TR/webnn/#api-mlgraphbuilder-flatten
 * https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/webnn/builders/impl/flatten_op_builder.cc
 */
export function Flatten(
  node: any,
  toJsVarName: (name: string) => string
): string {
  const inputVars = getInputVars(node, toJsVarName);
  const outputVars = getOutputVars(node, toJsVarName);
  const inputShape = getShape(node, 0);

  // Default axis is 1 (ONNX spec)
  let axis = 1;
  for (const attr of node.attributes || []) {
    if (attr.name === 'axis') {
      axis = typeof attr.value === 'number' ? attr.value : Number(attr.value?.value);
    }
  }
  const rank = inputShape.length;
  if (axis < 0) axis += rank;

  // Compute new shape
  const before = inputShape.slice(0, axis).reduce((a, b) => a * b, 1);
  const after = inputShape.slice(axis).reduce((a, b) => a * b, 1);

  const labelOpt = node.name ? `{ label: '${node.name}' }` : '';

  return `
    const ${outputVars[0]} = builder.reshape(
      ${inputVars[0]},
      [${before}, ${after}], 
      ${labelOpt}
    );
`;
}