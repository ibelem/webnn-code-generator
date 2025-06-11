import {
  getInputVars,
  getOutputVars,
  getAttr
} from '../../operation-utils';

/**
 * Generate JavaScript code for a WebNN instanceNormalization operation from ONNX InstanceNormalization node info.
 * Handles both NCHW and NHWC layouts.
 * https://www.w3.org/TR/webnn/#api-mlgraphbuilder-instancenorm
 * https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/webnn/builders/impl/normalization_op_builder.cc
 */
export function InstanceNormalization(
  node: any,
  toJsVarName: (name: string) => string,
  options: { nhwc?: boolean } = {}
): string {
  // ONNX: [input, scale, bias]
  // WebNN: instanceNormalization(input, {scale, bias, epsilon, layout, label})
  const inputVars = getInputVars(node, toJsVarName); // [input, scale, bias]
  const outputVars = getOutputVars(node, toJsVarName);

  const epsilon = getAttr(node, 'epsilon', 1e-5);

  // Determine layout: default to NCHW, but allow NHWC if requested
  const layout = options.nhwc ? 'nhwc' : 'nchw';

  let opts = [
    `scale: ${inputVars[1]}`,
    `bias: ${inputVars[2]}`,
    `epsilon: ${epsilon}`,
    `layout: '${layout}'`,
    `label: '${node.name}'`
  ];

  return `
    const ${outputVars[0]} = builder.instanceNormalization(
      ${inputVars[0]},
      { ${opts.join(', ')} }
    );
`;
}