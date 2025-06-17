import {
  getInputVars,
  getOutputVars,
  getShape,
  getAttrValue
} from '../../operation-utils';

/**
 * Generate JavaScript code for WebNN from ONNX Flatten node.
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
  let axis = getAttrValue(node, 'axis', 1);

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