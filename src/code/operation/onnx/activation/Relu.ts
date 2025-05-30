/**
 * Generate JavaScript code for a WebNN relu operation from ONNX Relu node info.
 * https://www.w3.org/TR/webnn/#api-mlgraphbuilder-relu-method
 */

import { getNonEmptyStringAroundNewline } from '../../../../utils';

export function Relu(
  node: any,
  toJsVarName: (name: string) => string
): string {
  const inputs: string[] = node.inputs?.map((i: any) => getNonEmptyStringAroundNewline(i.value?.[0]?.name)) || [];
  const outputs: string[] = node.outputs?.map((o: any) => getNonEmptyStringAroundNewline(o.value?.[0]?.name)) || [];
  const inputVar = toJsVarName(inputs[0]);
  const outputVar = toJsVarName(outputs[0]);

  return `
    const ${outputVar} = builder.relu(
      ${inputVar}
    );`;
}