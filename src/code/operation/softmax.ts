/**
 * Generate JavaScript code for a WebNN softmax operation from ONNX Softmax node info.
 * https://www.w3.org/TR/webnn/#api-mlgraphbuilder-softmax-method
 */

import { getNonEmptyStringAroundNewline } from '../../utils';

export function softmax_js(
  node: any,
  toJsVarName: (name: string) => string
): string {
  const inputs: string[] = node.inputs?.map((i: any) => getNonEmptyStringAroundNewline(i.value?.[0]?.name)) || [];
  const outputs: string[] = node.outputs?.map((o: any) => getNonEmptyStringAroundNewline(o.value?.[0]?.name)) || [];
  const attrs: any[] = node.attributes || [];
  const inputVar = toJsVarName(inputs[0]);
  const outputVar = toJsVarName(outputs[0]);

  // Default axis is 1 for ONNX Softmax
  let axis = 1;
  for (const attr of attrs) {
    if (attr.name === 'axis') {
      axis = Number(attr.i ?? 1);
      break;
    }
  }

  return `
    const ${outputVar} = builder.softmax(
      ${inputVar},
      ${axis}
    );`;
}