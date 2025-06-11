/**
 * Implements the ScaledDotProductAttention subgraph as a helper.
 * https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/webnn/builders/impl/attention_helper.h
 */
export function ScaledDotProductAttention(
  node: any,
  query: string,
  key: string,
  value: string,
  scale: string,
  attnMask: string | undefined,
  reshapeOutputShape: number[],
  outputVar: string
): string {
  const nodeName = node.name || '';
  let code = `
    // ScaledDotProductAttention subgraph
    const ${outputVar}_matmul1 = builder.matmul(
      ${query}, ${key},
      { label: '${nodeName}_/Attention/qkv/matmul_1' }
    );
    const ${outputVar}_scaled = builder.mul(
      ${outputVar}_matmul1, ${scale},
      { label: '${nodeName}_/Attention/qkv/div' }
    );`;

  let softmaxInput = `${outputVar}_scaled`;
  if (attnMask) {
    code += `
    const ${outputVar}_add_mask = builder.add(
      ${outputVar}_scaled, ${attnMask},
      { label: '${nodeName}_/Attention/attn_mask/softmax_input' }
    );`;
    softmaxInput = `${outputVar}_add_mask`;
  }

  code += `
    const ${outputVar}_softmax = builder.softmax(
      ${softmaxInput}, 3,
      { label: '${nodeName}_/Attention/attn_mask/softmax_input' }
    );
    const ${outputVar}_matmul2 = builder.matmul(
      ${outputVar}_softmax, ${value},
      { label: '${nodeName}_/Attention/qkv/matmul_2' }
    );
    const ${outputVar}_transposed = builder.transpose(
      ${outputVar}_matmul2,
      { permutation: [0,2,1,3], label: '${nodeName}_/Attention/qkv/transpose' }
    );
    const ${outputVar} = builder.reshape(
      ${outputVar}_transposed,
      [${reshapeOutputShape.join(', ')}],
      { label: '${nodeName}_/Attention/qkv/reshape' }
    );
`;
  return code;
}