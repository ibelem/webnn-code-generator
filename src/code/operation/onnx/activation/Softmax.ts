import {
  getInputVars,
  getOutputVars,
  getShape
} from '../../operation-utils';

/**
 * Generate JavaScript code for a WebNN softmax operation from ONNX Softmax node info.
 * https://www.w3.org/TR/webnn/#api-mlgraphbuilder-softmax-method
 */
export function Softmax(
  node: any,
  toJsVarName: (name: string) => string
): string {
  const inputVars = getInputVars(node, toJsVarName);
  const outputVars = getOutputVars(node, toJsVarName);
  const inputShape = getShape(node, 0);

  // Default axis is 1 for ONNX Softmax
  let axis = 1;
  for (const attr of node.attributes || []) {
    if (attr.name === 'axis') {
      if (attr.value && typeof attr.value.value === 'string') {
        axis = Number(attr.value.value);
      } else if (typeof attr.value === 'number') {
        axis = attr.value;
      }
      break;
    }
  }

  if (axis < 0) {
    axis = inputShape.length + axis;
  }

  return `
    const ${outputVars[0]} = builder.softmax(
      ${inputVars[0]},
      ${axis}
    );`;
}