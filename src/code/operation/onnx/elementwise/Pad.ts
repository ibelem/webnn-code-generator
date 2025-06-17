import {
  getInputVars,
  getOutputVars,
  getShape,
  getAttrValue
} from '../../operation-utils';

/**
 * Generate JavaScript code for a WebNN pad operation from ONNX Pad node info.
 * https://www.w3.org/TR/webnn/#api-mlgraphbuilder-pad
 * https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/webnn/builders/impl/pad_op_builder.cc
 */
export function Pad(
  node: any,
  toJsVarName: (name: string) => string
): string {
  const inputVars = getInputVars(node, toJsVarName);
  const outputVars = getOutputVars(node, toJsVarName);

  // Use getAttrValue for robust attribute extraction
  // ONNX default: mode='constant', value=0, pads=[]
  let mode = getAttrValue(node, 'mode', 'constant');
  let pads = getAttrValue(node, 'pads', []);
  let constantValue = getAttrValue(node, 'value', 0);

  // Map ONNX mode to WebNN mode
  const modeMap: Record<string, string> = {
    constant: 'constant',
    reflect: 'reflection',
    edge: 'edge'
  };
  mode = modeMap[String(mode).toLowerCase()] || 'constant';

  // ONNX pads: [begin_dim1, begin_dim2, ..., end_dim1, end_dim2, ...]
  // WebNN: beginningPadding, endingPadding (each of length N)
  let beginningPadding: number[] = [];
  let endingPadding: number[] = [];
  if (pads.length % 2 === 0 && pads.length > 0) {
    const half = pads.length / 2;
    beginningPadding = pads.slice(0, half);
    endingPadding = pads.slice(half);
  }

  // Clamp negative padding to zero for WebNN, and slice after if needed
  const clampNeg = (arr: number[]) => arr.map(x => Math.max(0, x));
  const hasNegative = (arr: number[]) => arr.some(x => x < 0);
  const clampedBegin = clampNeg(beginningPadding);
  const clampedEnd = clampNeg(endingPadding);
  const needsSlice = hasNegative(beginningPadding) || hasNegative(endingPadding);

  // WebNN pad options
  const padOpts: string[] = [];
  if (mode !== 'constant') padOpts.push(`mode: '${mode}'`);
  if (mode === 'constant' && constantValue !== 0) padOpts.push(`value: ${constantValue}`);
  if (node.name) padOpts.push(`label: '${node.name}'`);

  const optsString = padOpts.length ? `, { ${padOpts.join(', ')} }` : '';

  let code = `
    const ${outputVars[0]}_padded = builder.pad(
      ${inputVars[0]},
      [${clampedBegin.join(', ')}],
      [${clampedEnd.join(', ')}]${optsString}
    );`;

  // If negative padding, add a slice op after pad
  if (needsSlice && getShape(node, 0, false)) {
    const inputShape = getShape(node, 0, false);
    const starts = beginningPadding.map((v) => v < 0 ? -v : 0);
    const sizes = inputShape.map((dim: number, i: number) =>
      dim + (beginningPadding[i] || 0) + (endingPadding[i] || 0)
    );
    code += `
    const ${outputVars[0]} = builder.slice(
      ${outputVars[0]}_padded,
      [${starts.join(', ')}],
      [${sizes.join(', ')}],
      { label: '${node.name || ''}_slice_output' }
    );`;
  } else {
    code += `
    const ${outputVars[0]} = ${outputVars[0]}_padded;`;
  }

  return code;
}