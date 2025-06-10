import {
  getInputVars,
  getOutputVars
} from '../operation-utils';

/**
 * Generate JavaScript code for a WebNN cumulativeSum operation from ONNX CumSum node info.
 * https://www.w3.org/TR/webnn/#api-mlgraphbuilder-cumulativesum
 * https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/webnn/builders/impl/cumsum_op_builder.cc
 */
export function CumSum(
  node: any,
  toJsVarName: (name: string) => string
): string {
  const inputVars = getInputVars(node, toJsVarName);
  const outputVars = getOutputVars(node, toJsVarName);

  // ONNX axis is usually input[1] as a constant or attribute
  let axis = 0;
  let inputRank = node.inputs?.[0]?.shape?.length ?? 0;
  if (node.inputs.length > 1 && node.inputs[1]?.value?.[0]?.initializer) {
    // Extract axis from initializer
    const init = node.inputs[1].value[0].initializer;
    const arr = Object.keys(init.values)
      .sort((a, b) => Number(a) - Number(b))
      .map(k => init.values[k]);
    axis = Number(arr[0]);
    // Handle negative axis
    if (axis < 0 && inputRank > 0) {
      axis = inputRank + axis;
    }
  } else if (node.attributes) {
    for (const attr of node.attributes) {
      if (attr.name === 'axis') {
        axis = typeof attr.value === 'number' ? attr.value : Number(attr.value?.value);
        if (axis < 0 && inputRank > 0) {
          axis = inputRank + axis;
        }
        break;
      }
    }
  }

  // ONNX CumSum has exclusive and reverse attributes
  let exclusive = false;
  let reversed = false;
  for (const attr of node.attributes || []) {
    if (attr.name === 'exclusive') {
      exclusive = !!(typeof attr.value === 'number' ? attr.value : attr.value?.value);
    }
    if (attr.name === 'reverse') {
      reversed = !!(typeof attr.value === 'number' ? attr.value : attr.value?.value);
    }
  }

  // Add label option if node.name is present
  const labelOpt = node.name ? `label: '${node.name}'` : '';

  return `
    const ${outputVars[0]} = builder.cumulativeSum(
      ${inputVars[0]},
      ${axis},
      { exclusive: ${exclusive}, reversed: ${reversed}, ${labelOpt} }
    );`;
}