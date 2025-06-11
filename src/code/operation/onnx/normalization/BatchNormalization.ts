import {
  getInputVars,
  getOutputVars,
  getAttr
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

  // Try to get input shape if available
  const inputShape = node.inputs?.[0]?.shape;
  // Determine layout: default to NCHW, allow NHWC if requested
  const layout = options.nhwc ? 'nhwc' : 'nchw';

  // Axis: NCHW=1, NHWC=last dimension
  let axis = getAttr(node, 'axis', undefined);
  if (axis === undefined) {
    if (layout === 'nhwc' && inputShape && inputShape.length > 0) {
      axis = inputShape.length - 1;
    } else {
      axis = 1;
    }
  }

  const epsilon = getAttr(node, 'epsilon', 1e-5);

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
    );
`;
}