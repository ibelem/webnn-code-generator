/**
 * Generate JavaScript code for a WebNN binary operation (add, sub, mul, div, max, min, pow).
 * @param node - The ONNX node object (with inputs, outputs)
 * @param toJsVarName - Function to convert ONNX names to JS variable names
 * @param opType - The binary operation type (e.g. 'add', 'sub', 'mul', etc.)
 * @returns JavaScript code string for the binary operation
 */

/**
 * WebNN Specification: https://www.w3.org/TR/webnn/
 * https://www.w3.org/TR/webnn/#api-mlgraphbuilder-binary
 */

import { getNonEmptyStringAroundNewline } from '../../utils';
export function binary_js(
  node: any,
  toJsVarName: (name: string) => string,
  opType: string
): string {
  // Extract input and output names
  const inputs: string[] = node.inputs?.map((i: any) => getNonEmptyStringAroundNewline(i.value?.[0].name)) || [];
  const outputs: string[] = node.outputs?.map((o: any) => getNonEmptyStringAroundNewline(o.value?.[0].name)) || [];

  const inputVars = inputs.map(toJsVarName);
  const outputVar = toJsVarName(outputs[0]);

  return `
    const ${outputVar} = builder.${opType}(
      ${inputVars[0]},
      ${inputVars[1]}
    );`;
}