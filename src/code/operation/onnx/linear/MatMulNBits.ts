import {
  getInputVars,
  getOutputVars,
  getAttr
} from '../../operation-utils';

/**
 * Decompose ONNX MatMulNBits op into WebNN ops: dequantizeLinear + transpose + matmul (+ add bias).
 * https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/webnn/builders/impl/matMulNBits_op_builder.cc
 * Only supports 4-bit quantization (uint4).
 */
export function MatMulNBits(
  node: any,
  toJsVarName: (name: string) => string
): string {
  const inputVars = getInputVars(node, toJsVarName); // [A, B, scales, zero_points, g_idx, bias]
  const outputVars = getOutputVars(node, toJsVarName);

  // Get attributes
  const K = getAttr(node, 'K', 0);
  const N = getAttr(node, 'N', 0);

  // Shapes for dequantization
  // B: [N, n_blocks_per_col, blob_size] (uint8) → [N, n_blocks_per_col, blob_size*2] (uint4)
  // scales: [N, n_blocks_per_col] → [N, n_blocks_per_col, 1]
  // zero_points: [N, n_blocks_per_col] → [N, n_blocks_per_col, 1]
  // For simplicity, assume n_blocks_per_col and blob_size are known from input shapes

  // We'll use pseudo-code for shape extraction, as actual shape info may not be available at codegen time
  const n_blocks_per_col = `/*n_blocks_per_col*/`;
  const double_blob_size = `/*blob_size*2*/`;

  // DequantizeLinear: x = B (uint4), x_scale = scales, x_zero_point = zero_points or default 8
  let code = `
    // == ${node.name} · start ==
    // Decompose ${node.name} into WebNN ops
    /* Dequantize B (uint4) using scales and zero_points */
    const ${outputVars[0]}_B_uint4 = builder.constant(
      {type: 'uint4', shape: [${N}, ${n_blocks_per_col}, ${double_blob_size}]},
      /* TODO: Provide uint4 data for B */
    );
    const ${outputVars[0]}_scales_reshaped = builder.reshape(
      ${inputVars[2]},
      [${N}, ${n_blocks_per_col}, 1],
      {label: '${node.name}_reshape_scales'}
    );
`;

  if (inputVars[3]) {
    code += `
    const ${outputVars[0]}_zero_points_reshaped = builder.reshape(
      ${inputVars[3]},
      [${N}, ${n_blocks_per_col}, 1],
      {label: '${node.name}_reshape_zero_points'}
    );
`;
  } else {
    code += `
    const ${outputVars[0]}_zero_points_reshaped = builder.constant(
      {type: 'uint4', shape: [${N}, ${n_blocks_per_col}, 1]},
      new Uint8Array([8]) // default zero_point = 8
    );
`;
  }

  code += `
    const ${outputVars[0]}_dequantized = builder.dequantizeLinear(
      ${outputVars[0]}_B_uint4,
      ${outputVars[0]}_scales_reshaped,
      ${outputVars[0]}_zero_points_reshaped,
      {label: '${node.name}_dequantizeLinear'}
    );
    // Reshape to [N, K]
    const ${outputVars[0]}_reshaped = builder.reshape(
      ${outputVars[0]}_dequantized,
      [${N}, ${K}],
      {label: '${node.name}_reshape_dequantized'}
    );
    // Transpose to [K, N]
    const ${outputVars[0]}_transposed = builder.transpose(
      ${outputVars[0]}_reshaped,
      {label: '${node.name}_transpose'}
    );
    // MatMul
    let ${outputVars[0]} = builder.matmul(
      ${inputVars[0]},
      ${outputVars[0]}_transposed,
      {label: '${node.name}_matmul'}
    );
`;

  // Optional bias add
  if (inputVars[5]) {
    code += `
    ${outputVars[0]} = builder.add(
      ${outputVars[0]},
      ${inputVars[5]},
      {label: '${node.name}_add_bias'}
    );
    // == ${node.name} · end ==
`;
  }

  return code;
}