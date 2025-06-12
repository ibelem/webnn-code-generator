import {
  getInputVars,
  getOutputVars,
  getAttr
} from '../../operation-utils';

/**
 * Decompose ONNX RotaryEmbedding op into a series of WebNN ops.
 * Implements rotary embedding as described in ONNXRuntime and DML EP.
 * See: https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/webnn/builders/impl/rotaryEmbedding_op_builder.cc
 */
export function RotaryEmbedding(
  node: any,
  toJsVarName: (name: string) => string,
  options: { [key: string]: any } = {}
): string {
  const nhwc = !!options.nhwc;

  // ONNX: [input, cos_cache, sin_cache, position_ids?]
  // WebNN: not supported directly, decompose using split, gather, mul, add, concat, reshape, etc.
  const inputVars = getInputVars(node, toJsVarName); // [input, cos_cache, sin_cache, position_ids?]
  const outputVars = getOutputVars(node, toJsVarName);

  // Attributes
  const interleaved = !!getAttr(node, 'interleaved', 0);
  let num_heads = getAttr(node, 'num_heads', 0);
  let rotary_embedding_dim = getAttr(node, 'rotary_embedding_dim', 0);

  // Shapes
  const inputShape = node.inputs?.[0]?.shape || [];
  const cosCacheShape = node.inputs?.[1]?.shape || [];
  const hasPositionIds = !!node.inputs?.[3];
  // const positionIdsShape = node.inputs?.[3]?.shape || [];
  const inputIs4D = inputShape.length === 4;
  const batchSize = inputShape[0];
  let sequenceLength, hiddenSize, headSize;

  if (inputIs4D) {
    // NCHW: [batch, num_heads, seq, head_size]
    sequenceLength = nhwc ? inputShape[1] : inputShape[2];
    num_heads = nhwc ? inputShape[2] : inputShape[1];
    headSize = nhwc ? inputShape[3] : inputShape[3];
    hiddenSize = num_heads * headSize;
  } else {
    // [batch, seq, hidden]
    sequenceLength = inputShape[1];
    hiddenSize = inputShape[2];
    if (num_heads) {
      headSize = hiddenSize / num_heads;
    } else {
      headSize = cosCacheShape[1] * 2;
      num_heads = hiddenSize / headSize;
    }
  }
  if (!rotary_embedding_dim) rotary_embedding_dim = headSize;
  const halfRotaryDim = rotary_embedding_dim / 2;

  // 1. Reshape or transpose input to [batch, seq, num_heads, head_size]
  let code = '';
  code += `
    // == ${node.name} · start ==
    // Decompose ${node.name} into WebNN ops`;
  if (inputIs4D) {
    // If NHWC, assume input is already [batch, seq, num_heads, head_size]
    // If NCHW, transpose from [batch, num_heads, seq, head_size] -> [batch, seq, num_heads, head_size]
    if (nhwc) {
      code += `
    const ${outputVars[0]}_main = ${inputVars[0]};`;
    } else {
      code += `
    const ${outputVars[0]}_main = builder.transpose(
      ${inputVars[0]},
      { permutation: [0, 2, 1, 3], label: '${node.name}_transpose_input' }
    );`;
    }
  } else {
    code += `
    const ${outputVars[0]}_main = builder.reshape(
      ${inputVars[0]},
      [${batchSize}, ${sequenceLength}, ${num_heads}, ${headSize}],
      { label: '${node.name}_reshape_input' }
    );`;
  }
  const inputMain = `${outputVars[0]}_main`;

  // 2. Split input on last dim if rotary_embedding_dim < head_size
  let partial0 = inputMain;
  let partial1 = null;
  if (headSize !== rotary_embedding_dim) {
    code += `
    const [${outputVars[0]}_partial0, ${outputVars[0]}_partial1] = builder.split(
      ${inputMain},
      [${rotary_embedding_dim}, ${headSize - rotary_embedding_dim}],
      3,
      { label: '${node.name}_split_input' }
    );`;
    partial0 = `${outputVars[0]}_partial0`;
    partial1 = `${outputVars[0]}_partial1`;
  }

  // 3. Reshape partial0 for interleaved or not
  const partial0Shape = interleaved
    ? `[${batchSize}, ${sequenceLength}, ${num_heads}, ${halfRotaryDim}, 2]`
    : `[${batchSize}, ${sequenceLength}, ${num_heads}, 2, ${halfRotaryDim}]`;
  code += `
    const ${outputVars[0]}_partial0_reshaped = builder.reshape(
      ${partial0},
      ${partial0Shape},
      { label: '${node.name}_reshape_partial0' }
    );`;

  // 4. Split partial0 into 2 along axis (4 if interleaved, else 3)
  const splitAxis = interleaved ? 4 : 3;
  code += `
    const [${outputVars[0]}_a, ${outputVars[0]}_b] = builder.split(
      ${outputVars[0]}_partial0_reshaped,
      2,
      ${splitAxis},
      { label: '${node.name}_split_partial0' }
    );`;

  // 5. Reverse and concat for rotated part
  code += `
    const ${outputVars[0]}_rotated = builder.concat(
      [${outputVars[0]}_b, ${outputVars[0]}_a],
      ${splitAxis},
      { label: '${node.name}_concat_rotated' }
    );`;

  // 6. Gather cos/sin using position_ids if present
  let cosVar = inputVars[1];
  let sinVar = inputVars[2];
  if (hasPositionIds) {
    code += `
    const ${outputVars[0]}_gather_cos = builder.gather(
      ${inputVars[1]},
      ${inputVars[3]},
      { axis: 0, label: '${node.name}_gather_cos' }
    );
    const ${outputVars[0]}_gather_sin = builder.gather(
      ${inputVars[2]},
      ${inputVars[3]},
      { axis: 0, label: '${node.name}_gather_sin' }
    );`;
    cosVar = `${outputVars[0]}_gather_cos`;
    sinVar = `${outputVars[0]}_gather_sin`;
  }

  // 7. Reshape cos/sin for broadcast
  const cosSinShape = interleaved
    ? `[${batchSize}, ${sequenceLength}, 1, ${halfRotaryDim}, 1]`
    : `[${batchSize}, ${sequenceLength}, 1, 1, ${halfRotaryDim}]`;
  code += `
    const ${outputVars[0]}_cos_broadcast = builder.reshape(
      ${cosVar},
      ${cosSinShape},
      { label: '${node.name}_reshape_cos' }
    );
    const ${outputVars[0]}_sin_broadcast = builder.reshape(
      ${sinVar},
      ${cosSinShape},
      { label: '${node.name}_reshape_sin' }
    );`;

  // 8. Mul partial0 with cos, rotated with sin
  code += `
    const ${outputVars[0]}_mul_cos = builder.mul(
      ${outputVars[0]}_partial0_reshaped,
      ${outputVars[0]}_cos_broadcast,
      { label: '${node.name}_mul_cos' }
    );
    const ${outputVars[0]}_mul_sin = builder.mul(
      ${outputVars[0]}_rotated,
      ${outputVars[0]}_sin_broadcast,
      { label: '${node.name}_mul_sin' }
    );`;

  // 9. Create sign tensor and multiply with mul_sin
  const signShape = interleaved
    ? `[1, 1, 1, 2]`
    : `[1, 1, 2, 1]`;
  code += `
    const ${outputVars[0]}_sign = builder.constant(
      { type: 'float32', shape: ${signShape} },
      new Float32Array([-1, 1])
    );
    const ${outputVars[0]}_mul_sin_signed = builder.mul(
      ${outputVars[0]}_mul_sin,
      ${outputVars[0]}_sign,
      { label: '${node.name}_mul_sign' }
    );`;

  // 10. Reshape mul_cos and mul_sin_signed to [batch, seq, num_heads, rotary_embedding_dim]
  const rotaryShape = `[${batchSize}, ${sequenceLength}, ${num_heads}, ${rotary_embedding_dim}]`;
  code += `
    const ${outputVars[0]}_cos_final = builder.reshape(
      ${outputVars[0]}_mul_cos,
      ${rotaryShape},
      { label: '${node.name}_reshape_mul_cos' }
    );
    const ${outputVars[0]}_sin_final = builder.reshape(
      ${outputVars[0]}_mul_sin_signed,
      ${rotaryShape},
      { label: '${node.name}_reshape_mul_sin' }
    );`;

  // 11. Add cos and sin parts
  code += `
    const ${outputVars[0]}_rotary = builder.add(
      ${outputVars[0]}_cos_final,
      ${outputVars[0]}_sin_final,
      { label: '${node.name}_add_rotary' }
    );`;

  // 12. If split, concat with partial1
  let rotaryOut = `${outputVars[0]}_rotary`;
  if (partial1) {
    code += `
    const ${outputVars[0]}_joined = builder.concat(
      [${outputVars[0]}_rotary, ${partial1}],
      3,
      { label: '${node.name}_concat_final' }
    );`;
    rotaryOut = `${outputVars[0]}_joined`;
  }

  // 13. Transpose or reshape back to original shape
  if (inputIs4D) {
    // If NHWC, no need to transpose back
    if (nhwc) {
      code += `
    const ${outputVars[0]} = ${rotaryOut};`;
        } else {
          code += `
    const ${outputVars[0]} = builder.transpose(
      ${rotaryOut},
      { permutation: [0, 2, 1, 3], label: '${node.name}_transpose_output' }
    );`;
    }
  } else {
    code += `
    const ${outputVars[0]} = builder.reshape(
      ${rotaryOut},
      [${inputShape.join(', ')}],
      { label: '${node.name}_reshape_output' }
    );
    // == ${node.name} · end ==
`;
  }

  return code;
}