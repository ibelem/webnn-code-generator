import {
  getOutputVars,
  getShape,
  getAttrValue
} from '../../operation-utils';

/**
 * Generate JavaScript code for a WebNN equivalent of the ONNX Shape op using constant + slice workaround.
 * https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/webnn/builders/impl/shape_op_builder.cc
 */
export function Shape(
  node: any,
  toJsVarName: (name: string) => string,
  options: { [key: string]: any } = {}
): string {
  const nhwc = !!options.nhwc;
  const outputVars = getOutputVars(node, toJsVarName);
  const inputShape = getShape(node, 0, nhwc);

  // Prefer int64, fallback to int32 for compatibility
  // For codegen, use int32 for broad compatibility
  const dtype = 'int32';

  let start = getAttrValue(node, 'start', 0);
  let end = getAttrValue(node, 'end', inputShape.length);

  // Clamp and handle negatives
  const rank = inputShape.length;
  start = Math.max(0, start < 0 ? start + rank : start);
  end = Math.max(start, end < 0 ? end + rank : end);
  end = Math.min(end, rank);
  const sliceLength = end - start;

  // Add label for slice op
  const labelOpt = node.name ? `{ label: '${node.name}' }` : '';

  return `
    const ${outputVars[0]}_shapeConst = builder.constant(
      {type: '${dtype}', shape: [${rank}]},
      new Int32Array([${inputShape.join(', ')}])
    );
    const ${outputVars[0]} = builder.slice(
      ${outputVars[0]}_shapeConst,
      [${start}],
      [${sliceLength}], ${labelOpt}
    );
`;
}