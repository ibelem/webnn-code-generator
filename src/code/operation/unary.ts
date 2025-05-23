/**
 * Generate JavaScript code for a WebNN unary operation (abs, ceil, cos, exp, floor, log, neg, relu, round, sigmoid, sin, sqrt, tanh).
 * @param node - The ONNX node object (with inputs, outputs)
 * @param toJsVarName - Function to convert ONNX names to JS variable names
 * @param opType - The unary operation type (e.g. 'abs', 'relu', etc.)
 * @returns JavaScript code string for the unary operation
 */

/**
 * WebNN Specification: https://www.w3.org/TR/webnn/
 * https://www.w3.org/TR/webnn/#api-mlgraphbuilder-unary
 */

import { getNonEmptyStringAroundNewline } from '../../utils';

export function unary_js(
  node: any,
  toJsVarName: (name: string) => string,
  opType: string
): string {
  // Extract input and output names
  const inputs: string[] = node.inputs?.map((i: any) => getNonEmptyStringAroundNewline(i.value?.[0].name)) || [];
  const outputs: string[] = node.outputs?.map((o: any) => getNonEmptyStringAroundNewline(o.value?.[0].name)) || [];

  const inputVar = toJsVarName(inputs[0]);
  const outputVar = toJsVarName(outputs[0]);

  return `
    const ${outputVar} = builder.${opType}(
      ${inputVar}
    );`;
}