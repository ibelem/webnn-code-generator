/**
 * Generate JavaScript code for a WebNN transpose operation from ONNX Transpose node info.
 * https://www.w3.org/TR/webnn/#api-mlgraphbuilder-transpose
 */

import { getNonEmptyStringAroundNewline } from '../../utils';

export function transpose(
  node: any,
  toJsVarName: (name: string) => string
): string {
  const inputs: string[] = node.inputs?.map((i: any) => getNonEmptyStringAroundNewline(i.value?.[0]?.name)) || [];
  const outputs: string[] = node.outputs?.map((o: any) => getNonEmptyStringAroundNewline(o.value?.[0]?.name)) || [];
  const attrs: any[] = node.attributes || [];
  const inputVar = toJsVarName(inputs[0]);
  const outputVar = toJsVarName(outputs[0]);

  // Default perm is reversed order if not specified
  let perm: number[] | null = null;
  for (const attr of attrs) {
    if (attr.name === 'perm' && Array.isArray(attr.ints)) {
      perm = attr.ints;
      break;
    }
  }

  if (perm !== null) {
    return `
    const ${outputVar} = builder.transpose(
      ${inputVar},
      { permutation: [${perm.join(', ')}] }
    );`;
  } else {
    return `
    const ${outputVar} = builder.transpose(
      ${inputVar}
    );`;
  }
}