import {
  getInputVars,
  getOutputVars,
  getAttr
} from '../../operation-utils';

/**
 * Generate JavaScript code for a WebNN layerNormalization operation from ONNX LayerNormalization node info.
 * https://www.w3.org/TR/webnn/#api-mlgraphbuilder-layernorm
 * https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/webnn/builders/impl/normalization_op_builder.cc
 * 
 * Note: The axes parameter determines which dimensions to normalize over.
 * For typical normalization in NCHW, use axes=[1,2,3]. For NHWC, use axes=[1,2,3].
 * For NLP models with (batch,seq,features), typical normalization is axes=[2].
 */
export function LayerNormalization(
  node: any,
  toJsVarName: (name: string) => string,
  options: { nhwc?: boolean } = {}
): string {
  // ONNX: [input, scale, bias]
  // WebNN: layerNormalization(input, {scale, bias, axes, epsilon, label})
  const inputVars = getInputVars(node, toJsVarName); // [input, scale, bias]
  const outputVars = getOutputVars(node, toJsVarName);
  const nhwc = !!options.nhwc;

  // Get epsilon attribute, default 1e-5
  const epsilon = getAttr(node, 'epsilon', 1e-5);

  // Get axes attribute. ONNX spec says default is to normalize over last axis.
  let axes = getAttr(node, 'axes', undefined);
  
  if (!axes) {
    // Try to infer rank from input shape if available
    const shape = node.inputs?.[0]?.shape;
    if (shape && shape.length > 0) {
      // Default behavior: normalize over last axis (per ONNX spec)
      axes = [shape.length - 1];
    } else {
      // Fallback to last dimension if shape unknown
      axes = [-1];
    }
    
    // For 4D tensors (image data), detect if we should normalize over features based on layout
    if (shape && shape.length === 4) {
      // Common case for images: normalize over features
      // For NCHW: normalize over C (axis=1)
      // For NHWC: normalize over C (axis=3)
      if (nhwc) {
        axes = [3]; // NHWC - channels are last
      } else {
        axes = [1]; // NCHW - channels are second
      }
    }
  }

  // Convert negative axes to positive
  const positiveAxes = (axes as number[]).map((axis: number) => 
    axis < 0 ? `(${inputVars[0]}.shape.length + ${axis})` : axis
  );

  let opts = [
    `scale: ${inputVars[1]}`,
    `bias: ${inputVars[2]}`,
    `axes: [${positiveAxes.join(', ')}]`,
    `epsilon: ${epsilon}`
  ];
  
  // Only add label if node.name exists
  if (node.name) {
    opts.push(`label: '${node.name}'`);
  }

  return `
    // LayerNormalization - ${nhwc ? 'NHWC' : 'NCHW'} layout
    const ${outputVars[0]} = builder.layerNormalization(
      ${inputVars[0]},
      { ${opts.join(', ')} }
    );
`;
}