import {
  getInputVars,
  getOutputVars,
  getAttr
} from '../../operation-utils';

/**
 * Generate JavaScript code for a WebNN layerNormalization operation from ONNX LayerNormalization node info.
 * https://www.w3.org/TR/webnn/#api-mlgraphbuilder-layernorm
 * https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/webnn/builders/impl/normalization_op_builder.cc
 */
export function LayerNormalization(
  node: any,
  toJsVarName: (name: string) => string
): string {
  // ONNX: [input, scale, bias]
  // WebNN: layerNormalization(input, {scale, bias, axes, epsilon, label})
  const inputVars = getInputVars(node, toJsVarName); // [input, scale, bias]
  const outputVars = getOutputVars(node, toJsVarName);

  // Get epsilon attribute, default 1e-5
  const epsilon = getAttr(node, 'epsilon', 1e-5);

  // Get axes attribute, default to [1, 2, ..., rank-1] if not present
  let axes = getAttr(node, 'axes', undefined);
  if (!axes) {
    // Try to infer rank from input shape if available, else use [1]
    const shape = node.inputs?.[0]?.shape;
    if (shape && shape.length > 1) {
      axes = Array.from({length: shape.length - 1}, (_, i) => i + 1);
    } else {
      axes = [1];
    }
  }

  let opts = [
    `scale: ${inputVars[1]}`,
    `bias: ${inputVars[2]}`,
    `axes: [${axes.join(', ')}]`,
    `epsilon: ${epsilon}`,
    `label: '${node.name}'`
  ];

  return `
const ${outputVars[0]} = builder.layerNormalization(
  ${inputVars[0]},
  { ${opts.join(', ')} }
);
`;
}