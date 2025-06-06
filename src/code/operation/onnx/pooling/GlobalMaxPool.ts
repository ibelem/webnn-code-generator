import {
  getInputVars,
  getOutputVars,
  getShape
} from '../../operation-utils';

/**
 * Generate JavaScript code for a WebNN global maxPool2d operation from ONNX GlobalMaxPool node info.
 * https://www.w3.org/TR/webnn/#api-mlgraphbuilder-pool2d-max
 */
export function GlobalMaxPool(
  node: any,
  toJsVarName: (name: string) => string,
  options: { [key: string]: any } = {}
): string {
  const inputVars = getInputVars(node, toJsVarName);
  const outputVars = getOutputVars(node, toJsVarName);
  const nhwc = !!options.nhwc;

  const inputShape = getShape(node, 0, nhwc);  // Add nhwc parameter

  // Layout
  const layout = nhwc ? "'nhwc'" : "'nchw'";

  // For global pooling, windowDimensions is the spatial dims of the input
  // NCHW: [b, c, h, w] -> [h, w], NHWC: [b, h, w, c] -> [h, w]
  const windowDims = nhwc
    ? [inputShape[1], inputShape[2]]
    : [inputShape[2], inputShape[3]];

  return `
    const ${outputVars[0]} = builder.maxPool2d(
      ${inputVars[0]},
      {
        windowDimensions: [${windowDims.join(', ')}],
        layout: ${layout}
      }
    );`;
}