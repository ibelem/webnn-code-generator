import {
  getInputVars,
  getOutputVars,
  getAttr
} from '../../operation-utils';

/**
 * Decompose ONNX SimplifiedLayerNormalization op into a series of WebNN ops.
 * Implements: Y = scale * (X - mean) / sqrt(variance + epsilon)
 * where mean and variance are computed across the specified axes.
 * No bias is used in SimplifiedLayerNormalization.
 * https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/webnn/builders/impl/normalization_op_builder.cc
 */
export function SimplifiedLayerNormalization(
  node: any,
  toJsVarName: (name: string) => string
): string {
  // ONNX: [input, scale]
  // WebNN: not supported directly, decompose using add, div, mul, pow, reduceMean, sqrt
  const inputVars = getInputVars(node, toJsVarName); // [input, scale]
  const outputVars = getOutputVars(node, toJsVarName);

  // Get epsilon attribute, default 1e-5
  const epsilon = getAttr(node, 'epsilon', 1e-5);

  // Get axes attribute, default to [1, 2, ..., rank-1] if not present
  let axes = getAttr(node, 'axes', undefined);
  // Try to infer rank from input shape if available, else use [1]
  const shape = node.inputs?.[0]?.shape;
  if (!axes) {
    if (shape && shape.length > 1) {
      axes = Array.from({length: shape.length - 1}, (_, i) => i + 1);
    } else {
      axes = [1];
    }
  }

  // Compose code for decomposition
  return `
    // == ${node.name} · start ==
    // Decompose ${node.name} into WebNN ops
    const ${outputVars[0]}_mean = builder.reduceMean(
      ${inputVars[0]},
      { axes: [${axes.join(', ')}], keepDimensions: true, label: '${node.name}_reduceMean' }
    );
    const ${outputVars[0]}_centered = builder.sub(
      ${inputVars[0]},
      ${outputVars[0]}_mean,
      { label: '${node.name}_centered' }
    );
    const ${outputVars[0]}_squared = builder.pow(
      ${outputVars[0]}_centered,
      builder.constant({type: 'float32', shape: []}, new Float32Array([2])),
      { label: '${node.name}_pow2' }
    );
    const ${outputVars[0]}_var = builder.reduceMean(
      ${outputVars[0]}_squared,
      { axes: [${axes.join(', ')}], keepDimensions: true, label: '${node.name}_reduceVar' }
    );
    const ${outputVars[0]}_var_eps = builder.add(
      ${outputVars[0]}_var,
      builder.constant({type: 'float32', shape: []}, new Float32Array([${epsilon}])),
      { label: '${node.name}_add_epsilon' }
    );
    const ${outputVars[0]}_std = builder.sqrt(
      ${outputVars[0]}_var_eps,
      { label: '${node.name}_sqrt' }
    );
    const ${outputVars[0]}_norm = builder.div(
      ${outputVars[0]}_centered,
      ${outputVars[0]}_std,
      { label: '${node.name}_div' }
    );
    const ${outputVars[0]} = builder.mul(
      ${inputVars[1]},
      ${outputVars[0]}_norm,
      { label: '${node.name}_mul_scale' }
    );
    // == ${node.name} · end ==
`;
}