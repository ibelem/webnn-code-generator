/**
 * Generate JavaScript code for a WebNN add operation from ONNX Add node info.
 * @param node - The ONNX node object (with inputs, outputs)
 * @param toJsVarName - Function to convert ONNX names to JS variable names
 * @returns JavaScript code string for the add operation
 */

import {  getNonEmptyStringAroundNewline  } from '../../utils';
export function add_js(
  node: any,
  toJsVarName: (name: string) => string
): string {
  // Extract input and output names
  const inputs: string[] = node.inputs?.map((i: any) => getNonEmptyStringAroundNewline(i.value?.[0].name)) || [];
  const outputs: string[] = node.outputs?.map((o: any) => getNonEmptyStringAroundNewline(o.value?.[0].name)) || [];

  const inputVars = inputs.map(toJsVarName);
  const outputVar = toJsVarName(outputs[0]);

  return `
    const ${outputVar} = builder.add(${inputVars[0]}, ${inputVars[1]});`;
}