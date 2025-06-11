import {
  getInputVars,
  getOutputVars,
  getAttr
} from '../../operation-utils';

/**
 * Generate JavaScript code for a WebNN batchNormalization operation from ONNX BatchNormalization node info.
 * https://www.w3.org/TR/webnn/#api-mlgraphbuilder-batchnorm
 * https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/webnn/builders/impl/normalization_op_builder.cc
 */

export function BatchNormalization(
  node: any,
  toJsVarName: (name: string) => string
): string {
  // ONNX: [input, scale, bias, mean, var]
  // WebNN: batchNormalization(input, mean, variance, {scale, bias, axis, epsilon})
  const inputVars = getInputVars(node, toJsVarName); // [input, scale, bias, mean, var]
  const outputVars = getOutputVars(node, toJsVarName);

  // Default axis for NCHW is 1, for NHWC is usually last dim
  const axis = getAttr(node, 'axis', 1);
  const epsilon = getAttr(node, 'epsilon', 1e-5);

  // Compose options
  let options = [
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
      { ${options.join(', ')} }
    );
`;
}