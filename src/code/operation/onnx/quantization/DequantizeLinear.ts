import {
  getInputVars, getOutputVars, getShape, getDtype, inlineReshape, zeroConstant
} from '../../operation-utils';

/**
 * Generate JavaScript code for WebNN dequantizeLinear operation from ONNX node info.
 * https://www.w3.org/TR/webnn/#api-mlgraphbuilder-dequantizelinear
 */
export function DequantizeLinear(
  node: any,
  toJsVarName: (name: string) => string,
  options: { [key: string]: any } = {}
): string {
  const nhwc = !!options.nhwc;
  const inputVars = getInputVars(node, toJsVarName);
  const outputVars = getOutputVars(node, toJsVarName);
  const inputShape = getShape(node, 0, nhwc);  // Add nhwc parameter
  const scaleShape = getShape(node, 1, nhwc);  // Add nhwc parameter
  const inputDtype = getDtype(node, 0);

  let axis = 1;
  for (const attr of node.attributes || []) {
    if (attr.name === 'axis') {
      axis = typeof attr.value?.value === 'string' ? Number(attr.value.value) : attr.value;
      break;
    }
  }

  let scaleExpr = inlineReshape(inputVars[1], scaleShape, (() => {
    if (scaleShape.length === inputShape.length) return scaleShape;
    const newShape = Array(inputShape.length).fill(1);
    if (scaleShape.length === 1) newShape[axis] = scaleShape[0];
    return newShape;
  })());

  let zeroPointExpr: string;
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

  return `
    const ${outputVars[0]} = builder.dequantizeLinear(
      ${inputVars[0]},
      ${scaleExpr},
      ${zeroPointExpr}
    );`;
}