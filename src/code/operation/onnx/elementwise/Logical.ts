import {
  getInputVars,
  getOutputVars
} from '../../operation-utils';

/**
 * Generate JavaScript code for a WebNN logical operation
 * https://www.w3.org/TR/webnn/#api-mlgraphbuilder-logical
 * https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/webnn/builders/impl/binary_op_builder.cc
 */

function Logical(
  node: any,
  toJsVarName: (name: string) => string,
  options: { [key: string]: any } = {}
): string {
  const inputVars = getInputVars(node, toJsVarName);
  const outputVars = getOutputVars(node, toJsVarName);
  const opType = options.opType;
  const opts = node.name ? `{ label: '${node.name}' }` : '{}';

  // Check input count
  if ((opType === 'logicalNot' && inputVars.length !== 1) ||
      (opType !== 'logicalNot' && inputVars.length !== 2)) {
    return `// ERROR: ${opType} expects ${opType === 'logicalNot' ? 1 : 2} input(s), got ${inputVars.length}`;
  }

  if (opType === 'logicalNot') {
    return `
    const ${outputVars[0]} = builder.logicalNot(
      ${inputVars[0]},
      ${opts}
    );`;
  } else {
    return `
    const ${outputVars[0]} = builder.${opType}(
      ${inputVars[0]},
      ${inputVars[1]},
      ${opts}
    );`;
  }
}

// ONNX op to WebNN op mapping
export function Equal(node: any, toJsVarName: (name: string) => string, options: { [key: string]: any } = {}): string {
  return Logical(node, toJsVarName, { ...options, opType: 'equal' });
}
export function Greater(node: any, toJsVarName: (name: string) => string, options: { [key: string]: any } = {}): string {
  return Logical(node, toJsVarName, { ...options, opType: 'greater' });
}
export function GreaterOrEqual(node: any, toJsVarName: (name: string) => string, options: { [key: string]: any } = {}): string {
  return Logical(node, toJsVarName, { ...options, opType: 'greaterOrEqual' });
}
export function Less(node: any, toJsVarName: (name: string) => string, options: { [key: string]: any } = {}): string {
  return Logical(node, toJsVarName, { ...options, opType: 'lesser' });
}
export function LessOrEqual(node: any, toJsVarName: (name: string) => string, options: { [key: string]: any } = {}): string {
  return Logical(node, toJsVarName, { ...options, opType: 'lesserOrEqual' });
}
export function Not(node: any, toJsVarName: (name: string) => string, options: { [key: string]: any } = {}): string {
  return Logical(node, toJsVarName, { ...options, opType: 'logicalNot' });
}
export function And(node: any, toJsVarName: (name: string) => string, options: { [key: string]: any } = {}): string {
  return Logical(node, toJsVarName, { ...options, opType: 'logicalAnd' });
}
export function Or(node: any, toJsVarName: (name: string) => string, options: { [key: string]: any } = {}): string {
  return Logical(node, toJsVarName, { ...options, opType: 'logicalOr' });
}
export function Xor(node: any, toJsVarName: (name: string) => string, options: { [key: string]: any } = {}): string {
  return Logical(node, toJsVarName, { ...options, opType: 'logicalXor' });
}