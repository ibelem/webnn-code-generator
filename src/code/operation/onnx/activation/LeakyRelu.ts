/**
 * Generate JavaScript code for a WebNN leakyRelu operation from ONNX LeakyRelu node info.
 * https://www.w3.org/TR/webnn/#api-mlgraphbuilder-leakyrelu
 */

import { getNonEmptyStringAroundNewline } from '../../../../utils';

export function LeakyRelu(
  node: any,
  toJsVarName: (name: string) => string
): string {
  const inputs: string[] = node.inputs?.map((i: any) => getNonEmptyStringAroundNewline(i.value?.[0]?.name)) || [];
  const outputs: string[] = node.outputs?.map((o: any) => getNonEmptyStringAroundNewline(o.value?.[0]?.name)) || [];
  const attrs: any[] = node.attributes || [];
  const inputVar = toJsVarName(inputs[0]);
  const outputVar = toJsVarName(outputs[0]);

  // Default alpha is 0.01 for ONNX LeakyRelu
  let alpha = 0.01;
  for (const attr of attrs) {
    if (attr.name === 'alpha') {
      alpha = Number(attr.f ?? 0.01);
      break;
    }
  }

  return `
    const ${outputVar} = builder.leakyRelu(
      ${inputVar},
      { alpha: ${alpha} }
    );`;
}