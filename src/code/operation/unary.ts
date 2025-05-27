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

function unary(
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

export function abs(node: any, toJsVarName: (name: string) => string): string {
  return unary(node, toJsVarName, 'abs');
}
export function ceil(node: any, toJsVarName: (name: string) => string): string {
  return unary(node, toJsVarName, 'ceil');
}
export function cos(node: any, toJsVarName: (name: string) => string): string {
  return unary(node, toJsVarName, 'cos');
}
export function erf(node: any, toJsVarName: (name: string) => string): string {
  return unary(node, toJsVarName, 'erf');
}
export function exp(node: any, toJsVarName: (name: string) => string): string {
  return unary(node, toJsVarName, 'exp');
}
export function floor(node: any, toJsVarName: (name: string) => string): string {
  return unary(node, toJsVarName, 'floor');
}
export function identity(node: any, toJsVarName: (name: string) => string): string {
  return unary(node, toJsVarName, 'identity');
}
export function log(node: any, toJsVarName: (name: string) => string): string {
  return unary(node, toJsVarName, 'log');
}
export function neg(node: any, toJsVarName: (name: string) => string): string {
  return unary(node, toJsVarName, 'neg');
}
export function round(node: any, toJsVarName: (name: string) => string): string {
  return unary(node, toJsVarName, 'round');
}
export function reciprocal(node: any, toJsVarName: (name: string) => string): string {
  return unary(node, toJsVarName, 'reciprocal');
}
export function sin(node: any, toJsVarName: (name: string) => string): string {
  return unary(node, toJsVarName, 'sin');
}
export function sign(node: any, toJsVarName: (name: string) => string): string {
  return unary(node, toJsVarName, 'sign');
}
export function sqrt(node: any, toJsVarName: (name: string) => string): string {
  return unary(node, toJsVarName, 'sqrt');
}
export function tan(node: any, toJsVarName: (name: string) => string): string {
  return unary(node, toJsVarName, 'tan');
}