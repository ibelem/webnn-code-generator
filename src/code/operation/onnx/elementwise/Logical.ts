import { getNonEmptyStringAroundNewline } from '../../../../utils';

function Logical(
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

export function Equal(node: any, toJsVarName: (name: string) => string): string {
  return Logical(node, toJsVarName, 'equal');
}
export function Greater(node: any, toJsVarName: (name: string) => string): string {
  return Logical(node, toJsVarName, 'greater');
}
export function GreaterOrEqual(node: any, toJsVarName: (name: string) => string): string {
  return Logical(node, toJsVarName, 'greaterOrEqual');
}
export function Less(node: any, toJsVarName: (name: string) => string): string {
  return Logical(node, toJsVarName, 'lesser');
}
export function LessOrEqual(node: any, toJsVarName: (name: string) => string): string {
  return Logical(node, toJsVarName, 'lesserOrEqual');
}
export function Not(node: any, toJsVarName: (name: string) => string): string {
  return Logical(node, toJsVarName, 'logicalNot');
}
export function And(node: any, toJsVarName: (name: string) => string): string {
  return Logical(node, toJsVarName, 'logicalAnd');
}
export function Or(node: any, toJsVarName: (name: string) => string): string {
  return Logical(node, toJsVarName, 'logicalOr');
}
export function Xor(node: any, toJsVarName: (name: string) => string): string {
  return Logical(node, toJsVarName, 'logicalXor');
}