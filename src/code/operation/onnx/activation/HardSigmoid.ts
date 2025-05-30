/**
 * Generate JavaScript code for a WebNN hardSigmoid operation from ONNX HardSigmoid node info.
 * https://www.w3.org/TR/webnn/#api-mlgraphbuilder-hard-sigmoid
 */

import { getNonEmptyStringAroundNewline } from '../../../../utils';

export function HardSigmoid(
  node: any,
  toJsVarName: (name: string) => string
): string {
  const inputs: string[] = node.inputs?.map((i: any) => getNonEmptyStringAroundNewline(i.value?.[0]?.name)) || [];
  const outputs: string[] = node.outputs?.map((o: any) => getNonEmptyStringAroundNewline(o.value?.[0]?.name)) || [];
  const attrs: any[] = node.attributes || [];
  const inputVar = toJsVarName(inputs[0]);
  const outputVar = toJsVarName(outputs[0]);

  // Default values per ONNX spec: alpha=0.2, beta=0.5
  let alpha = 0.2;
  let beta = 0.5;
  for (const attr of attrs) {
    if (attr.name === 'alpha') alpha = Number(attr.f ?? 0.2);
    if (attr.name === 'beta') beta = Number(attr.f ?? 0.5);
  }

  return `
    const ${outputVar} = builder.hardSigmoid(
      ${inputVar},
      { alpha: ${alpha}, beta: ${beta} }
    );`;
}