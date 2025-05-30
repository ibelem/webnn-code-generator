import {
  getInputVars, getOutputVars, getShape, inlineReshape, zeroConstant
} from '../../operation-utils';

/**
 * Generate JavaScript code for WebNN quantizeLinear operation from ONNX node info.
 * https://www.w3.org/TR/webnn/#api-mlgraphbuilder-quantizelinear
 */
export function QuantizeLinear(
  node: any,
  toJsVarName: (name: string) => string,
  _options: { [key: string]: any } = {}
): string {
  const inputVars = getInputVars(node, toJsVarName);
  const outputVars = getOutputVars(node, toJsVarName);
  const inputShape = getShape(node, 0);
  const scaleShape = getShape(node, 1);
  const outputDtype = node.outputs?.[0]?.value?.[0]?.type?.dataType || 'uint8';

  // Axis
  let axis = 1;
  for (const attr of node.attributes || []) {
    if (attr.name === 'axis') {
      axis = typeof attr.value?.value === 'string' ? Number(attr.value.value) : attr.value;
      break;
    }
  }

  // Inline reshape for scale
  let scaleExpr = inlineReshape(inputVars[1], scaleShape, (() => {
    if (scaleShape.length === inputShape.length) return scaleShape;
    const newShape = Array(inputShape.length).fill(1);
    if (scaleShape.length === 1) newShape[axis] = scaleShape[0];
    return newShape;
  })());

  // Inline reshape or constant for zeroPoint
  let zeroPointExpr: string;
  if (node.inputs.length > 2 && node.inputs[2]?.value?.[0]) {
    const zpShape = getShape(node, 2);
    zeroPointExpr = inlineReshape(inputVars[2], zpShape, (() => {
      if (zpShape.length === inputShape.length) return zpShape;
      const newShape = Array(inputShape.length).fill(1);
      if (zpShape.length === 1) newShape[axis] = zpShape[0];
      return newShape;
    })());
  } else {
    // If zeroPoint is not present, create a constant zero of same shape as scale (after expansion)
    const zpShape = scaleShape.length === inputShape.length
      ? scaleShape
      : (() => {
          const arr = Array(inputShape.length).fill(1);
          if (scaleShape.length === 1) arr[axis] = scaleShape[0];
          return arr;
        })();
    zeroPointExpr = zeroConstant(outputDtype, zpShape);
  }

  return `
    const ${outputVars[0]} = builder.quantizeLinear(
      ${inputVars[0]},
      ${scaleExpr},
      ${zeroPointExpr}
    );`;
}