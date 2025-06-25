import {
  getInputVars,
  getOutputVars,
  getAttrValue,
  getShape
} from '../../operation-utils';

/**
 * Generate JavaScript code for a WebNN batchNormalization operation from ONNX BatchNormalization node info.
 * Handles both NCHW and NHWC layouts by setting the axis accordingly.
 * https://www.w3.org/TR/webnn/#api-mlgraphbuilder-batchnorm
 * https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/webnn/builders/impl/normalization_op_builder.cc
 */

export function BatchNormalization(
  node: any,
  toJsVarName: (name: string) => string,
  options: { nhwc?: boolean } = {}
): string {
  // ONNX: [input, scale, bias, mean, var]
  // WebNN: batchNormalization(input, mean, variance, {scale, bias, axis, epsilon, label})
  const inputVars = getInputVars(node, toJsVarName); // [input, scale, bias, mean, var]
  const outputVars = getOutputVars(node, toJsVarName);
  const nhwc = !!options.nhwc;

  // Try to get input shape if available
  const { shape: inputShape } = getShape(node, 0, nhwc);
  // Determine layout: default to NCHW, allow NHWC if requested
  // Axis: NCHW=1, NHWC=last dimension
  let axis = getAttrValue(node, 'axis', undefined);
  if (axis === undefined) {
    if (nhwc && inputShape && inputShape.length > 0) {
      axis = inputShape.length - 1;
    } else {
      axis = 1;
    }
  }

  const epsilon = getAttrValue(node, 'epsilon', 1e-5);

  // Compose options
  let opts = [
    `scale: ${inputVars[1]}`,
    `bias: ${inputVars[2]}`,
    `axis: ${axis}`,
    `epsilon: ${epsilon}`,
    `label: '${node.name}'`
  ];

  return `
    const ${outputVars[0]} = builder.batchNormalization(
      ${inputVars[0]}, // input
      ${inputVars[3]}, // mean
      ${inputVars[4]}, // variance
      { ${opts.join(', ')} }
    );`;
}