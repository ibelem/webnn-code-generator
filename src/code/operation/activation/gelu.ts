/**
 * Generate JavaScript code for a WebNN gelu operation from ONNX Gelu node info.
 * https://www.w3.org/TR/webnn/#api-mlgraphbuilder-gelu-method
 */

import { getNonEmptyStringAroundNewline } from '../../../utils';

export function gelu(
  node: any,
  toJsVarName: (name: string) => string
): string {
  const inputs: string[] = node.inputs?.map((i: any) => getNonEmptyStringAroundNewline(i.value?.[0]?.name)) || [];
  const outputs: string[] = node.outputs?.map((o: any) => getNonEmptyStringAroundNewline(o.value?.[0]?.name)) || [];
  const attrs: any[] = node.attributes || [];
  const inputVar = toJsVarName(inputs[0]);
  const outputVar = toJsVarName(outputs[0]);

  // Default mode is 'exact' for ONNX Gelu
  let mode = 'exact';
  for (const attr of attrs) {
    if (attr.name === 'approximate') {
      mode = attr.s === 'tanh' ? 'tanh' : 'exact';
      break;
    }
  }

  return `
    const ${outputVar} = builder.gelu(
      ${inputVar},
      { mode: '${mode}' }
    );`;
}