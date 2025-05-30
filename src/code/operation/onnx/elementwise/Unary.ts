import {
  getInputVars,
  getOutputVars
} from '../../operation-utils';

/**
 * Generate JavaScript code for a WebNN unary operation (abs, ceil, cos, exp, floor, log, neg, relu, round, sigmoid, sin, sqrt, tanh).
 * @param node - The ONNX node object (with inputs, outputs)
 * @param toJsVarName - Function to convert ONNX names to JS variable names
 * @param options - Options object, must include opType
 * @returns JavaScript code string for the unary operation
 */
function Unary(
  node: any,
  toJsVarName: (name: string) => string,
  options: { [key: string]: any } = {}
): string {
  const inputVars = getInputVars(node, toJsVarName);
  const outputVars = getOutputVars(node, toJsVarName);
  const opType = options.opType;

  return `
    const ${outputVars[0]} = builder.${opType}(
      ${inputVars[0]}
    );`;
}

export function Abs(node: any, toJsVarName: (name: string) => string, options: { [key: string]: any } = {}): string {
  return Unary(node, toJsVarName, { ...options, opType: 'abs' });
}
export function Ceil(node: any, toJsVarName: (name: string) => string, options: { [key: string]: any } = {}): string {
  return Unary(node, toJsVarName, { ...options, opType: 'ceil' });
}
export function Cos(node: any, toJsVarName: (name: string) => string, options: { [key: string]: any } = {}): string {
  return Unary(node, toJsVarName, { ...options, opType: 'cos' });
}
export function Erf(node: any, toJsVarName: (name: string) => string, options: { [key: string]: any } = {}): string {
  return Unary(node, toJsVarName, { ...options, opType: 'erf' });
}
export function Exp(node: any, toJsVarName: (name: string) => string, options: { [key: string]: any } = {}): string {
  return Unary(node, toJsVarName, { ...options, opType: 'exp' });
}
export function Floor(node: any, toJsVarName: (name: string) => string, options: { [key: string]: any } = {}): string {
  return Unary(node, toJsVarName, { ...options, opType: 'floor' });
}
export function Identity(node: any, toJsVarName: (name: string) => string, options: { [key: string]: any } = {}): string {
  return Unary(node, toJsVarName, { ...options, opType: 'identity' });
}
export function Log(node: any, toJsVarName: (name: string) => string, options: { [key: string]: any } = {}): string {
  return Unary(node, toJsVarName, { ...options, opType: 'log' });
}
export function Neg(node: any, toJsVarName: (name: string) => string, options: { [key: string]: any } = {}): string {
  return Unary(node, toJsVarName, { ...options, opType: 'neg' });
}
export function Round(node: any, toJsVarName: (name: string) => string, options: { [key: string]: any } = {}): string {
  return Unary(node, toJsVarName, { ...options, opType: 'round' });
}
export function Reciprocal(node: any, toJsVarName: (name: string) => string, options: { [key: string]: any } = {}): string {
  return Unary(node, toJsVarName, { ...options, opType: 'reciprocal' });
}
export function Sin(node: any, toJsVarName: (name: string) => string, options: { [key: string]: any } = {}): string {
  return Unary(node, toJsVarName, { ...options, opType: 'sin' });
}
export function Sign(node: any, toJsVarName: (name: string) => string, options: { [key: string]: any } = {}): string {
  return Unary(node, toJsVarName, { ...options, opType: 'sign' });
}
export function Sqrt(node: any, toJsVarName: (name: string) => string, options: { [key: string]: any } = {}): string {
  return Unary(node, toJsVarName, { ...options, opType: 'sqrt' });
}
export function Tan(node: any, toJsVarName: (name: string) => string, options: { [key: string]: any } = {}): string {
  return Unary(node, toJsVarName, { ...options, opType: 'tan' });
}