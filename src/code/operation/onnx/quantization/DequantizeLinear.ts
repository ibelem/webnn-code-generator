import {
  getInputVars, getOutputVars, getShape, getDtype, inlineReshape, zeroConstant
} from '../../operation-utils';

/**
 * Generate JavaScript code for WebNN dequantizeLinear operation from ONNX node info.
 * https://www.w3.org/TR/webnn/#api-mlgraphbuilder-dequantizelinear
 * https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/webnn/builders/impl/qdq_op_builder.cc
 */
export function DequantizeLinear(
  node: any,
  toJsVarName: (name: string) => string,
  options: { [key: string]: any } = {}
): string {
  const nhwc = !!options.nhwc;
  const inputVars = getInputVars(node, toJsVarName);
  const outputVars = getOutputVars(node, toJsVarName);
  const inputShape = getShape(node, 0, nhwc);
  const scaleShape = getShape(node, 1, nhwc);
  const inputDtype = getDtype(node, 0);

  // Axis attribute (default 1, handle negative axis)
  let axis = 1;
  for (const attr of node.attributes || []) {
    if (attr.name === 'axis') {
      axis = typeof attr.value?.value === 'string' ? Number(attr.value.value) : attr.value;
      break;
    }
  }
  if (axis < 0) {
    axis = inputShape.length + axis;
  }

  // Reshape scale and zero_point for per-axis quantization if needed
  let scaleExpr = inputVars[1];
  let zeroPointExpr: string;

  // Per-axis: scale is 1D, input is higher rank, and axis is not last
  if (
    scaleShape.length === 1 &&
    inputShape.length > 1 &&
    axis !== inputShape.length - 1
  ) {
    const targetShape = [
      ...Array(axis).fill(1),
      scaleShape[0],
      ...Array(inputShape.length - axis - 1).fill(1)
    ];
    scaleExpr = `builder.reshape(${inputVars[1]}, [${targetShape.join(', ')}], { label: '${node.name ?? ''}_reshape_scale' })`;

    if (node.inputs.length > 2 && node.inputs[2]?.value?.[0]) {
      zeroPointExpr = `builder.reshape(${inputVars[2]}, [${targetShape.join(', ')}], { label: '${node.name ?? ''}_reshape_zero_point' })`;
    } else {
      zeroPointExpr = zeroConstant(inputDtype, targetShape);
    }
  } else {
    // No special reshape needed
    scaleExpr = inlineReshape(inputVars[1], scaleShape, (() => {
      if (scaleShape.length === inputShape.length) return scaleShape;
      const newShape = Array(inputShape.length).fill(1);
      if (scaleShape.length === 1) newShape[axis] = scaleShape[0];
      return newShape;
    })());

    if (node.inputs.length > 2 && node.inputs[2]?.value?.[0]) {
      const zpShape = getShape(node, 2, nhwc);
      zeroPointExpr = inlineReshape(inputVars[2], zpShape, (() => {
        if (zpShape.length === inputShape.length) return zpShape;
        const newShape = Array(inputShape.length).fill(1);
        if (zpShape.length === 1) newShape[axis] = zpShape[0];
        return newShape;
      })());
    } else {
      const zpShape = scaleShape.length === inputShape.length
        ? scaleShape
        : (() => {
            const arr = Array(inputShape.length).fill(1);
            if (scaleShape.length === 1) arr[axis] = scaleShape[0];
            return arr;
          })();
      zeroPointExpr = zeroConstant(inputDtype, zpShape);
    }
  }

  const labelOpt = node.name ? `{ label: '${node.name}' }` : '{}';

  return `
    const ${outputVars[0]} = builder.dequantizeLinear(
      ${inputVars[0]},
      ${scaleExpr},
      ${zeroPointExpr},
      ${labelOpt}
    );`;
}