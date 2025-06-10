import {
  getInputVars,
  getOutputVars
} from '../operation-utils';

/**
 * Generate JavaScript code for a WebNN concat operation from ONNX Concat node info.
 * https://www.w3.org/TR/webnn/#api-mlgraphbuilder-concat
 * https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/webnn/builders/impl/concat_op_builder.cc
 */
export function Concat(
  node: any,
  toJsVarName: (name: string) => string
): string {
  const inputVars = getInputVars(node, toJsVarName);
  const outputVars = getOutputVars(node, toJsVarName);

  // Find axis and handle negative axis
  let axis = 0;
  let inputRank = node.inputs?.[0]?.shape?.length ?? 0;
  for (const attr of node.attributes || []) {
    if (attr.name === 'axis') {
      let val = typeof attr.value === 'number' ? attr.value : attr.value?.value;
      axis = Number(val);
      if (axis < 0 && inputRank > 0) {
        axis = inputRank + axis;
      }
      break;
    }
  }

  // Add label option if node.name is present
  const labelOpt = node.name ? `{ label: '${node.name}' }` : '';

  return `
    const ${outputVars[0]} = builder.concat(
      [${inputVars.join(', ')}],
      ${axis},
      ${labelOpt}
    );`;
}