/**
 * Generate JavaScript code for a WebNN softplus operation from ONNX Softplus node info.
 * https://www.w3.org/TR/webnn/#api-mlgraphbuilder-softplus-method
 */

import { getNonEmptyStringAroundNewline } from '../../../../utils';

export function Softplus(
  node: any,
  toJsVarName: (name: string) => string
): string {
  const inputs: string[] = node.inputs?.map((i: any) => getNonEmptyStringAroundNewline(i.value?.[0]?.name)) || [];
  const outputs: string[] = node.outputs?.map((o: any) => getNonEmptyStringAroundNewline(o.value?.[0]?.name)) || [];
  const inputVar = toJsVarName(inputs[0]);
  const outputVar = toJsVarName(outputs[0]);

  return `
    const ${outputVar} = builder.softplus(
      ${inputVar}
    );`;
}