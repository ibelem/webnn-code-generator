import {
  getInputVars,
  getOutputVars
} from '../operation-utils';

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

  // Default values
  let mode = 'constant';
  let value = 0;
  let pads: number[] = [];
  let constantValue = 0;

  // Parse ONNX attributes
  for (const attr of node.attributes || []) {
    if (attr.name === 'mode') {
      // Map ONNX mode to WebNN mode
      const modeMap: Record<string, string> = {
        constant: 'constant',
        reflect: 'reflection',
        edge: 'edge'
      };
      const raw = typeof attr.value === 'string' ? attr.value : (attr.value?.value ?? 'constant');
      mode = modeMap[raw.toLowerCase()] || 'constant';
    }
    if (attr.name === 'pads') {
      pads = Array.isArray(attr.value) ? attr.value : attr.value?.value ?? [];
    }
    if (attr.name === 'value') {
      constantValue = typeof attr.value === 'number' ? attr.value : Number(attr.value?.value ?? 0);
    }
  }

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
  if (mode === 'constant' && (constantValue !== 0 || value !== 0)) padOpts.push(`value: ${constantValue || value}`);
  if (node.name) padOpts.push(`label: '${node.name}'`);

  const optsString = padOpts.length ? `, { ${padOpts.join(', ')} }` : '';

  let code = `
    const ${outputVars[0]}_padded = builder.pad(
      ${inputVars[0]},
      [${clampedBegin.join(', ')}],
      [${clampedEnd.join(', ')}]${optsString}
    );`;

  // If negative padding, add a slice op after pad
  if (needsSlice && node.inputs?.[0]?.shape) {
    const inputShape = node.inputs[0].shape;
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