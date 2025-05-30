/**
 * Generate JavaScript code for a WebNN sigmoid operation from ONNX Sigmoid node info.
 * https://www.w3.org/TR/webnn/#api-mlgraphbuilder-sigmoid-method
 */

import { getNonEmptyStringAroundNewline } from '../../../../utils';

export function Sigmoid(
  node: any,
  toJsVarName: (name: string) => string
): string {
  const inputs: string[] = node.inputs?.map((i: any) => getNonEmptyStringAroundNewline(i.value?.[0]?.name)) || [];
  const outputs: string[] = node.outputs?.map((o: any) => getNonEmptyStringAroundNewline(o.value?.[0]?.name)) || [];
  const inputVar = toJsVarName(inputs[0]);
  const outputVar = toJsVarName(outputs[0]);

  return `
    const ${outputVar} = builder.sigmoid(
      ${inputVar}
    );`;
}