import {
  getInputVars,
  getOutputVars
} from '../../operation-utils';
import { ScaledDotProductAttention } from './ScaledDotProductAttention';

/**
 * Implements ONNX MultiHeadAttention as a subgraph of WebNN ops.
 * This is a decomposition, not a single WebNN op.
 * https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/webnn/builders/impl/mha_op_builder.cc
 */
export function MultiHeadAttention(
  node: any,
  toJsVarName: (name: string) => string
): string {
  const inputVars = getInputVars(node, toJsVarName);
  const outputVars = getOutputVars(node, toJsVarName);

  // Extract attributes
  const attrs = node.attributes || [];
  const attrDict: Record<string, any> = {};
  for (const attr of attrs) attrDict[attr.name] = attr.value;

  const numHeads = Number(attrDict['num_heads'] || 0);
  if (!numHeads) throw new Error('MultiHeadAttention: num_heads attribute is required');

  // Shapes
  const qShape = node.inputs?.[0]?.shape || [];
  const batch = qShape[0], seq = qShape[1], hidden = qShape[2];
  const headSize = Math.floor(hidden / numHeads);

  // Reshape/transpose query, key, value
  const qReshaped = `${outputVars[0]}_q_reshape`;
  const qTrans = `${outputVars[0]}_q_trans`;
  const kReshaped = `${outputVars[0]}_k_reshape`;
  const kTrans = `${outputVars[0]}_k_trans`;
  const vReshaped = `${outputVars[0]}_v_reshape`;
  const vTrans = `${outputVars[0]}_v_trans`;

  let code = `
    // == ${node.name} · start ==
    // Decompose ${node.name} into WebNN ops
    const ${qReshaped} = builder.reshape(
      ${inputVars[0]},
      [${batch}, ${seq}, ${numHeads}, ${headSize}],
      { label: '${node.name}_/MHA/query/reshape' }
    );
    const ${qTrans} = builder.transpose(
      ${qReshaped},
      { permutation: [0,2,1,3], label: '${node.name}_/MHA/query/transpose' }
    );
    const ${kReshaped} = builder.reshape(
      ${inputVars[1]},
      [${batch}, ${seq}, ${numHeads}, ${headSize}],
      { label: '${node.name}_/MHA/key/reshape_1' }
    );
    const ${kTrans} = builder.transpose(
      ${kReshaped},
      { permutation: [0,2,1,3], label: '${node.name}_/MHA/key/transpose' }
    );
    const ${vReshaped} = builder.reshape(
      ${inputVars[2]},
      [${batch}, ${seq}, ${numHeads}, ${headSize}],
      { label: '${node.name}_/MHA/value/reshape_1' }
    );
    const ${vTrans} = builder.transpose(
      ${vReshaped},
      { permutation: [0,2,1,3], label: '${node.name}_/MHA/value/transpose' }
    );
    `;

  // Scale
  const scale = attrDict['scale'] || (1 / Math.sqrt(headSize));
  const scaleConst = `${outputVars[0]}_scale`;
  code += `
    const ${scaleConst} = builder.constant(
      { type: 'float32', shape: [1] },
      new Float32Array([${scale}])
    );
`;

  // Attention mask (optional, input 5)
  let attnMaskVar = undefined;
  if (inputVars[5]) {
    attnMaskVar = inputVars[5];
  }

  // Output shape after attention: [batch, seq, hidden]
  const reshapeOutputShape = [batch, seq, hidden];

  // Call ScaledDotProductAttention
  code += ScaledDotProductAttention(
    node,
    qTrans, kTrans, vTrans,
    scaleConst,
    attnMaskVar,
    reshapeOutputShape,
    outputVars[0]
  );

  code += `
    // == ${node.name} · end ==
`;

  return code;
}