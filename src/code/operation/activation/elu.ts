/**
 * Generate JavaScript code for a WebNN elu operation from ONNX Elu node info.
 * https://www.w3.org/TR/webnn/#api-mlgraphbuilder-elu
 */

import { getNonEmptyStringAroundNewline } from '../../../utils';

export function elu_js(
  node: any,
  toJsVarName: (name: string) => string
): string {
  const inputs: string[] = node.inputs?.map((i: any) => getNonEmptyStringAroundNewline(i.value?.[0]?.name)) || [];
  const outputs: string[] = node.outputs?.map((o: any) => getNonEmptyStringAroundNewline(o.value?.[0]?.name)) || [];
  const attrs: any[] = node.attributes || [];
  const inputVar = toJsVarName(inputs[0]);
  const outputVar = toJsVarName(outputs[0]);

  // Default alpha is 1.0 for ONNX Elu
  let alpha = 1.0;
  for (const attr of attrs) {
    if (attr.name === 'alpha') {
      alpha = Number(attr.f ?? 1.0);
      break;
    }
  }

  return `
    const ${outputVar} = builder.elu(
      ${inputVar},
      { alpha: ${alpha} }
    );`;
}