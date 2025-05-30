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

import { getNonEmptyStringAroundNewline } from '../../../../utils';

function Unary(
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

export function Abs(node: any, toJsVarName: (name: string) => string): string {
  return Unary(node, toJsVarName, 'abs');
}
export function Ceil(node: any, toJsVarName: (name: string) => string): string {
  return Unary(node, toJsVarName, 'ceil');
}
export function Cos(node: any, toJsVarName: (name: string) => string): string {
  return Unary(node, toJsVarName, 'cos');
}
export function Erf(node: any, toJsVarName: (name: string) => string): string {
  return Unary(node, toJsVarName, 'erf');
}
export function Exp(node: any, toJsVarName: (name: string) => string): string {
  return Unary(node, toJsVarName, 'exp');
}
export function Floor(node: any, toJsVarName: (name: string) => string): string {
  return Unary(node, toJsVarName, 'floor');
}
export function Identity(node: any, toJsVarName: (name: string) => string): string {
  return Unary(node, toJsVarName, 'identity');
}
export function Log(node: any, toJsVarName: (name: string) => string): string {
  return Unary(node, toJsVarName, 'log');
}
export function Neg(node: any, toJsVarName: (name: string) => string): string {
  return Unary(node, toJsVarName, 'neg');
}
export function Round(node: any, toJsVarName: (name: string) => string): string {
  return Unary(node, toJsVarName, 'round');
}
export function Reciprocal(node: any, toJsVarName: (name: string) => string): string {
  return Unary(node, toJsVarName, 'reciprocal');
}
export function Sin(node: any, toJsVarName: (name: string) => string): string {
  return Unary(node, toJsVarName, 'sin');
}
export function Sign(node: any, toJsVarName: (name: string) => string): string {
  return Unary(node, toJsVarName, 'sign');
}
export function Sqrt(node: any, toJsVarName: (name: string) => string): string {
  return Unary(node, toJsVarName, 'sqrt');
}
export function Tan(node: any, toJsVarName: (name: string) => string): string {
  return Unary(node, toJsVarName, 'tan');
}