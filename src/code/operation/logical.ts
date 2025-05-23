/**
 * Generate JavaScript code for a WebNN logical operation (equal, notEqual, greater, greaterOrEqual, lesser, lesserOrEqual, logicalAnd, logicalOr, logicalXor, logicalNot).
 * @param node - The ONNX node object (with inputs, outputs)
 * @param toJsVarName - Function to convert ONNX names to JS variable names
 * @param opType - The logical operation type (e.g. 'equal', 'logicalAnd', etc.)
 * @returns JavaScript code string for the logical operation
 */

/**
 * WebNN Specification: https://www.w3.org/TR/webnn/
 * https://www.w3.org/TR/webnn/#api-mlgraphbuilder-logical
 */

import { getNonEmptyStringAroundNewline } from '../../utils';

export function logical_js(
  node: any,
  toJsVarName: (name: string) => string,
  opType: string
): string {
  // Extract input and output names
  const inputs: string[] = node.inputs?.map((i: any) => getNonEmptyStringAroundNewline(i.value?.[0].name)) || [];
  const outputs: string[] = node.outputs?.map((o: any) => getNonEmptyStringAroundNewline(o.value?.[0].name)) || [];

  const inputVars = inputs.map(toJsVarName);
  const outputVar = toJsVarName(outputs[0]);

  // logicalNot is unary, others are binary
  if (opType === 'logicalNot') {
    return `
    const ${outputVar} = builder.logicalNot(
      ${inputVars[0]}
    );`;
  } else {
    return `
    const ${outputVar} = builder.${opType}(
      ${inputVars[0]},
      ${inputVars[1]}
    );`;
  }
}