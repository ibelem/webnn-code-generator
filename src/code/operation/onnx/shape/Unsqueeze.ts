import {
  getInputVars,
  getOutputVars,
  getShape,
  getAttrValue
} from '../../operation-utils';

/**
 * Generate JavaScript code for WebNN from ONNX Unsqueeze node.
 * Uses reshape to implement unsqueeze.
 * https://www.w3.org/TR/webnn/#api-mlgraphbuilder-unsqueeze
 * https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/webnn/builders/impl/squeeze_unsqueeze_op_builder.cc
 */
export function Unsqueeze(
  node: any,
  toJsVarName: (name: string) => string
): string {
  const inputVars = getInputVars(node, toJsVarName);
  const outputVars = getOutputVars(node, toJsVarName);
  const { shape: inputShape } = getShape(node, 0, false);

  // Get axes from input[1] (opset >= 13) or attribute
  let axes: number[] = [];
  if (node.inputs.length > 1 && node.inputs[1]?.value?.[0]?.initializer) {
    const init = node.inputs[1].value[0].initializer;
    axes = Object.keys(init.values)
      .sort((a, b) => Number(a) - Number(b))
      .map(k => Number(init.values[k]));
  } else if (node.attributes) {
    axes = getAttrValue(node, 'axes', undefined);
  }

  // Insert 1 at each axis (ascending order)
  let newShape = inputShape.slice();
  const rank = newShape.length + axes.length;
  const axesNorm = axes.map(a => (a < 0 ? a + rank : a)).sort((a, b) => a - b);
  for (const axis of axesNorm) {
    newShape.splice(axis, 0, 1);
  }

  const labelOpt = node.name ? `{ label: '${node.name}' }` : '';

  return `
    const ${outputVars[0]} = builder.reshape(
      ${inputVars[0]},
      [${newShape.join(', ')}],
      ${labelOpt}
    );`;
}