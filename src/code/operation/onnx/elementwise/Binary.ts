import {
  getInputVars,
  getOutputVars
} from '../../operation-utils';

/**
 * Generate JavaScript code for a WebNN binary operation (add, sub, mul, div, max, min, pow, prelu).
 * https://www.w3.org/TR/webnn/#api-mlgraphbuilder-binary
 * https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/webnn/builders/impl/binary_op_builder.cc
 */
function Binary(
  node: any,
  toJsVarName: (name: string) => string,
  options: { [key: string]: any } = {}
): string {
  const inputVars = getInputVars(node, toJsVarName);
  const outputVars = getOutputVars(node, toJsVarName);
  const opType = options.opType;

  // Add label for debugging
  const opts = node.name ? `{ label: '${node.name}' }` : '';

  return `
    const ${outputVars[0]} = builder.${opType}(
      ${inputVars[0]},
      ${inputVars[1]},
      ${opts}
    );`;
}

export function Add(node: any, toJsVarName: (name: string) => string, options: { [key: string]: any } = {}): string {
  return Binary(node, toJsVarName, { ...options, opType: 'add' });
}
export function Sub(node: any, toJsVarName: (name: string) => string, options: { [key: string]: any } = {}): string {
  return Binary(node, toJsVarName, { ...options, opType: 'sub' });
}
export function Mul(node: any, toJsVarName: (name: string) => string, options: { [key: string]: any } = {}): string {
  return Binary(node, toJsVarName, { ...options, opType: 'mul' });
}
export function Div(node: any, toJsVarName: (name: string) => string, options: { [key: string]: any } = {}): string {
  return Binary(node, toJsVarName, { ...options, opType: 'div' });
}
export function Max(node: any, toJsVarName: (name: string) => string, options: { [key: string]: any } = {}): string {
  return Binary(node, toJsVarName, { ...options, opType: 'max' });
}
export function Min(node: any, toJsVarName: (name: string) => string, options: { [key: string]: any } = {}): string {
  return Binary(node, toJsVarName, { ...options, opType: 'min' });
}
export function Pow(node: any, toJsVarName: (name: string) => string, options: { [key: string]: any } = {}): string {
  return Binary(node, toJsVarName, { ...options, opType: 'pow' });
}
