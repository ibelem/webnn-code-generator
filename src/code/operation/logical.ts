import { getNonEmptyStringAroundNewline } from '../../utils';

function logical(
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

export function equal(node: any, toJsVarName: (name: string) => string): string {
  return logical(node, toJsVarName, 'equal');
}
export function greater(node: any, toJsVarName: (name: string) => string): string {
  return logical(node, toJsVarName, 'greater');
}
export function greaterOrEqual(node: any, toJsVarName: (name: string) => string): string {
  return logical(node, toJsVarName, 'greaterOrEqual');
}
export function less(node: any, toJsVarName: (name: string) => string): string {
  return logical(node, toJsVarName, 'lesser');
}
export function lessOrEqual(node: any, toJsVarName: (name: string) => string): string {
  return logical(node, toJsVarName, 'lesserOrEqual');
}
export function not(node: any, toJsVarName: (name: string) => string): string {
  return logical(node, toJsVarName, 'logicalNot');
}
export function and(node: any, toJsVarName: (name: string) => string): string {
  return logical(node, toJsVarName, 'logicalAnd');
}
export function or(node: any, toJsVarName: (name: string) => string): string {
  return logical(node, toJsVarName, 'logicalOr');
}
export function xor(node: any, toJsVarName: (name: string) => string): string {
  return logical(node, toJsVarName, 'logicalXor');
}