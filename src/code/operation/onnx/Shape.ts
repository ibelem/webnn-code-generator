import {
  getOutputVars,
  getShape
} from '../operation-utils';

/**
 * Generate JavaScript code for a WebNN equivalent of the ONNX Shape op using constant + slice workaround.
 * This creates a constant tensor of the input's shape, then slices it according to optional start/end attributes.
 * https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/webnn/builders/impl/shape_op_builder.cc
 */
export function Shape(
  node: any,
  toJsVarName: (name: string) => string
): string {
  // const inputVars = getInputVars(node, toJsVarName);
  const outputVars = getOutputVars(node, toJsVarName);
  const inputShape = getShape(node, 0);

  // Determine dtype: prefer int64, fallback to int32 if not supported
  // For codegen, we'll use int32 for compatibility
  const dtype = 'int32';

  // Get start/end attributes if present (ONNX Shape-15+)
  let start = 0;
  let end = inputShape.length;
  for (const attr of node.attributes || []) {
    if (attr.name === 'start') {
      start = typeof attr.value === 'number' ? attr.value : Number(attr.value?.value);
    }
    if (attr.name === 'end') {
      end = typeof attr.value === 'number' ? attr.value : Number(attr.value?.value);
    }
  }
  // Clamp and handle negatives
  const rank = inputShape.length;
  start = Math.max(0, start < 0 ? start + rank : start);
  end = Math.max(start, end < 0 ? end + rank : end);
  end = Math.min(end, rank);
  const sliceLength = end - start;

  return `
    // ONNX Shape op emulated using constant + slice in WebNN
    const ${outputVars[0]}_shapeConst = builder.constant(
      {type: '${dtype}', shape: [${rank}]},
      new Int32Array([${inputShape.join(', ')}])
    );
    const ${outputVars[0]} = builder.slice(
      ${outputVars[0]}_shapeConst,
      [${start}],
      [${sliceLength}]
    );
  `;
}