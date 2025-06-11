import {
  getInputVars,
  getOutputVars
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
  let axis = 0;
  for (const attr of node.attributes || []) {
    if (attr.name === 'axis') {
      axis = typeof attr.value === 'number' ? attr.value : Number(attr.value?.value);
    }
    if (attr.name === 'reduction' && attr.value !== 'none') {
      throw new Error('WebNN scatterElements only supports reduction type "none" (default).');
    }
  }

  // Handle negative axis
  const inputShape = node.inputs?.[0]?.shape || [];
  const rank = inputShape.length;
  if (axis < 0) axis += rank;

  const labelOpt = node.name ? `{ axis: ${axis}, label: '${node.name}' }` : `{ axis: ${axis} }`;

  return `
    const ${outputVars[0]} = builder.scatterElements(
      ${inputVars[0]},
      ${inputVars[1]},
      ${inputVars[2]},
      ${labelOpt}
    );
`;
}