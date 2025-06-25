import {
  getInputVars,
  getOutputVars,
  getShape,
  getAttrValue
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
  const { shape } = getShape(node, 0, nhwc);

  // Default axis is 1 for ONNX Softmax (opset <13), -1 for opset >=13
  const initAxis = (typeof node.opset === 'number' && node.opset >= 13) ? -1 : 1;
  let axis = getAttrValue(node, 'axis', initAxis);

  // Handle negative axis
  if (axis < 0) {
    axis = shape.length + axis;
  }

  return `
    const ${outputVars[0]} = builder.softmax(
      ${inputVars[0]},
      ${axis},
      { label: '${node.name || ''}' }
    );`;
}