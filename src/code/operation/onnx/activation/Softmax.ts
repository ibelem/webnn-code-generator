import {
  getInputVars,
  getOutputVars,
  getShape
} from '../../operation-utils';

/**
 * Generate JavaScript code for a WebNN softmax operation from ONNX Softmax node info.
 * https://www.w3.org/TR/webnn/#api-mlgraphbuilder-softmax-method
 * https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/webnn/builders/impl/softmax_op_builder.cc
 */

export function Softmax(
  node: any,
  toJsVarName: (name: string) => string,
  options: { [key: string]: any } = {}
): string {
  const nhwc = !!options.nhwc;
  const inputVars = getInputVars(node, toJsVarName);
  const outputVars = getOutputVars(node, toJsVarName);
  const inputShape = getShape(node, 0, nhwc);

  // Default axis is 1 for ONNX Softmax (opset <13), -1 for opset >=13
  let axis = (typeof node.opset === 'number' && node.opset >= 13) ? -1 : 1;
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

  // Handle negative axis
  if (axis < 0) {
    axis = inputShape.length + axis;
  }

  return `
    const ${outputVars[0]} = builder.softmax(
      ${inputVars[0]},
      ${axis},
      { label: '${node.name || ''}' }
    );
`;
}