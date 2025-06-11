import {
  getInputVars,
  getOutputVars,
  getShape
} from '../../operation-utils';

/**
 * Generate JavaScript code for WebNN from ONNX Squeeze node.
 * https://www.w3.org/TR/webnn/#api-mlgraphbuilder-squeeze
 * https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/webnn/builders/impl/squeeze_unsqueeze_op_builder.cc
 */
export function Squeeze(
  node: any,
  toJsVarName: (name: string) => string
): string {
  const inputVars = getInputVars(node, toJsVarName);
  const outputVars = getOutputVars(node, toJsVarName);
  const inputShape = getShape(node, 0);

  // Get axes from input[1] (opset >= 13) or attribute
  let axes: number[] | undefined;
  if (node.inputs.length > 1 && node.inputs[1]?.value?.[0]?.initializer) {
    const init = node.inputs[1].value[0].initializer;
    axes = Object.keys(init.values)
      .sort((a, b) => Number(a) - Number(b))
      .map(k => Number(init.values[k]));
  } else if (node.attributes) {
    for (const attr of node.attributes) {
      if (attr.name === 'axes') {
        if (Array.isArray(attr.value)) {
          axes = attr.value;
          // Todo
        } else if (attr.value?.ints) {
          axes = attr.value.ints;
        } else if (attr.value?.value) {
          axes = Array.isArray(attr.value.value) ? attr.value.value : [attr.value.value];
        }
      }
    }
  }

  // Remove axes in descending order
  let newShape = inputShape.slice();
  if (axes && axes.length > 0) {
    // Handle negative axes
    const rank = inputShape.length;
    const axesNorm = axes.map(a => (a < 0 ? a + rank : a)).sort((a, b) => b - a);
    for (const axis of axesNorm) {
      if (newShape[axis] === 1) {
        newShape.splice(axis, 1);
      }
    }
  } else {
    // Remove all single-dim axes
    newShape = newShape.filter(d => d !== 1);
  }

  const labelOpt = node.name ? `{ label: '${node.name}' }` : '';

  return `
    const ${outputVars[0]} = builder.reshape(
      ${inputVars[0]},
      [${newShape.join(', ')}],
      ${labelOpt}
    );
`;
}