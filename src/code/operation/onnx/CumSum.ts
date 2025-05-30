import {
  getInputVars,
  getOutputVars
} from '../operation-utils';

/**
 * Generate JavaScript code for a WebNN cumulativeSum operation from ONNX CumSum node info.
 * https://www.w3.org/TR/webnn/#api-mlgraphbuilder-cumulativesum
 */
export function CumSum(
  node: any,
  toJsVarName: (name: string) => string
): string {
  const inputVars = getInputVars(node, toJsVarName);
  const outputVars = getOutputVars(node, toJsVarName);

  // ONNX axis is usually input[1] as a constant or attribute
  let axis = 0;
  if (node.inputs.length > 1 && node.inputs[1]?.value?.[0]?.initializer) {
    // Extract axis from initializer
    const init = node.inputs[1].value[0].initializer;
    const arr = Object.keys(init.values)
      .sort((a, b) => Number(a) - Number(b))
      .map(k => init.values[k]);
    axis = Number(arr[0]);
  } else if (node.attributes) {
    for (const attr of node.attributes) {
      if (attr.name === 'axis') {
        axis = typeof attr.value === 'number' ? attr.value : Number(attr.value?.value);
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

  return `
    const ${outputVars[0]} = builder.cumulativeSum(
      ${inputVars[0]},
      ${axis},
      { exclusive: ${exclusive}, reversed: ${reversed} }
    );`;
}