/**
 * Generate JavaScript code for a WebNN prelu operation from ONNX PRelu node info.
 * https://www.w3.org/TR/webnn/#api-mlgraphbuilder-prelu
 */

import { getNonEmptyStringAroundNewline } from '../../../../utils';

export function PRelu(
  node: any,
  toJsVarName: (name: string) => string
): string {
  const inputs: string[] = node.inputs?.map((i: any) => getNonEmptyStringAroundNewline(i.value?.[0]?.name)) || [];
  const outputs: string[] = node.outputs?.map((o: any) => getNonEmptyStringAroundNewline(o.value?.[0]?.name)) || [];
  // PRelu expects two inputs: input and slope
  const inputVar = toJsVarName(inputs[0]);
  const slopeVar = toJsVarName(inputs[1]);
  const outputVar = toJsVarName(outputs[0]);

  return `
    const ${outputVar} = builder.prelu(
      ${inputVar},
      ${slopeVar}
    );`;
}