import {
  getInputVars,
  getOutputVars,
  getAttr
} from '../../operation-utils';

/**
 * Decompose ONNX SimplifiedLayerNormalization op into a series of WebNN ops.
 * Implements: Y = scale * (X - mean) / sqrt(variance + epsilon)
 * No bias is used in SimplifiedLayerNormalization.
 * 
 * Note: This operation is layout-sensitive as normalization axes differ between NCHW and NHWC.
 * For 4D tensors in NCHW format, normalization is typically over axes=[1] (channels).
 * For 4D tensors in NHWC format, normalization is typically over axes=[3] (channels).
 */
export function SimplifiedLayerNormalization(
  node: any,
  toJsVarName: (name: string) => string,
  options: { nhwc?: boolean } = {}
): string {
  // ONNX: [input, scale]
  // WebNN: not supported directly, decompose using div, mul, pow, reduceMean, sqrt
  const inputVars = getInputVars(node, toJsVarName); // [input, scale]
  const outputVars = getOutputVars(node, toJsVarName);
  const nhwc = !!options.nhwc;

  // Get epsilon attribute, default 1e-5
  const epsilon = getAttr(node, 'epsilon', 1e-5);

  // Get axes attribute
  let axes = getAttr(node, 'axes', undefined);
  const shape = node.inputs?.[0]?.shape;
  
  if (!axes) {
    // Handle layout-specific axis selection for 4D tensors (image data)
    if (shape && shape.length === 4) {
      if (nhwc) {
        // For NHWC, normalize over channels (axis=3)
        axes = [3];
      } else {
        // For NCHW, normalize over channels (axis=1)
        axes = [1];
      }
    } else {
      // Default: normalize over feature dimensions (all but batch)
      if (shape && shape.length > 1) {
        axes = Array.from({length: shape.length - 1}, (_, i) => i + 1);
      } else {
        // Fallback if shape is unknown
        axes = [1];
      }
    }
  }

  // Compose code for decomposition
  return `
    // == ${node.name} · start ==
    // Decompose ${node.name} into WebNN ops (${nhwc ? 'NHWC' : 'NCHW'} layout)
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