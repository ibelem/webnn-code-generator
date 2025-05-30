import {
  getInputVars,
  getOutputVars
} from '../operation-utils';

/**
 * Generate JavaScript code for a WebNN pad operation from ONNX Pad node info.
 * https://www.w3.org/TR/webnn/#api-mlgraphbuilder-pad
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
      mode = typeof attr.value === 'string' ? attr.value.toLowerCase() : (attr.value?.value?.toLowerCase() ?? 'constant');
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

  // WebNN pad options
  const padOpts: string[] = [];
  if (mode !== 'constant') padOpts.push(`mode: '${mode}'`);
  if (mode === 'constant' && (constantValue !== 0 || value !== 0)) padOpts.push(`value: ${constantValue || value}`);

  const optsString = padOpts.length ? `, { ${padOpts.join(', ')} }` : '';

  return `
    const ${outputVars[0]} = builder.pad(
      ${inputVars[0]},
      [${beginningPadding.join(', ')}],
      [${endingPadding.join(', ')}]${optsString}
    );`;
}