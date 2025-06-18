import {
  getInputVars,
  getOutputVars,
  getAttrValue,
  getShape
} from '../../operation-utils';

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

  // Use getAttrValue for robust attribute extraction
  let axis = getAttrValue(node, 'axis', 0);

  // Try to get input rank for negative axis handling
  let inputRank = 0;
  if (node.inputs && node.inputs.length > 0) {
    const { shape } = getShape(node, 0, false);
    if (Array.isArray(shape)) inputRank = shape.length;
  }

  // Handle negative axis
  if (axis < 0 && inputRank > 0) {
    axis = inputRank + axis;
  }

  const opts = [`axis: ${axis}`];
  if (node.name) opts.push(`label: '${node.name}'`);

  return `
    const ${outputVars[0]} = builder.concat(
      [${inputVars.join(', ')}],
      { ${opts.join(', ')} }
    );
  `;
}