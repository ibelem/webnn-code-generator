import {
  getInputVars,
  getOutputVars
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
  let axis = 0;
  for (const attr of node.attributes || []) {
    if (attr.name === 'axis') {
      axis = typeof attr.value === 'number' ? attr.value : Number(attr.value?.value);
    }
  }

  // Handle negative axis
  const inputShape = node.inputs?.[0]?.shape || [];
  const rank = inputShape.length;
  if (axis < 0) axis += rank;

  const labelOpt = node.name ? `{ axis: ${axis}, label: '${node.name}' }` : `{ axis: ${axis} }`;

  return `
const ${outputVars[0]} = builder.gatherElements(
  ${inputVars[0]},
  ${inputVars[1]},
  ${labelOpt}
);
`;
}