import {
  getInputVars,
  getOutputVars,
  getShape,
  getAttrValue
} from '../../operation-utils';

/**
 * Generate JavaScript code for a WebNN scatterElements operation from ONNX ScatterElements node info.
 * https://www.w3.org/TR/webnn/#api-mlgraphbuilder-scatterelements
 * https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/webnn/builders/impl/scatterElements_op_builder.cc
 * Only supports reduction='none' (default).
 */
export function ScatterElements(
  node: any,
  toJsVarName: (name: string) => string
): string {
  const inputVars = getInputVars(node, toJsVarName); // [data, indices, updates]
  const outputVars = getOutputVars(node, toJsVarName);

  // Default axis is 0
  let axis = getAttrValue(node, 'axis', 0);
  // WebNN scatterElements only supports reduction type "none" (default).

  // Handle negative axis
  const { shape: inputShape } = getShape(node, 0, false);
  const rank = inputShape.length;
  if (axis < 0) axis += rank;

  const labelOpt = node.name ? `{ axis: ${axis}, label: '${node.name}' }` : `{ axis: ${axis} }`;

  return `
    const ${outputVars[0]} = builder.scatterElements(
      ${inputVars[0]},
      ${inputVars[1]},
      ${inputVars[2]},
      ${labelOpt}
    );`;
}