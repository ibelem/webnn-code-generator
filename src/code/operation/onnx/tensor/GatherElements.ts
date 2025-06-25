import {
  getInputVars,
  getOutputVars,
  getShape,
  getAttrValue
} from '../../operation-utils';

/**
 * Generate JavaScript code for a WebNN gatherElements operation from ONNX GatherElements node info.
 * https://www.w3.org/TR/webnn/#api-mlgraphbuilder-gatherelements
 * https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/webnn/builders/impl/gatherElements_op_builder.cc
 */
export function GatherElements(
  node: any,
  toJsVarName: (name: string) => string
): string {
  const inputVars = getInputVars(node, toJsVarName);
  const outputVars = getOutputVars(node, toJsVarName);

  // Default axis is 0
  let axis = getAttrValue(node, 'axis', 0);

  // Handle negative axis
  const { shape: inputShape } = getShape(node, 0, false);
  const rank = inputShape.length;
  if (axis < 0) axis += rank;

  const labelOpt = node.name ? `{ axis: ${axis}, label: '${node.name}' }` : `{ axis: ${axis} }`;

  return `
const ${outputVars[0]} = builder.gatherElements(
  ${inputVars[0]},
  ${inputVars[1]},
  ${labelOpt}
);`;
}