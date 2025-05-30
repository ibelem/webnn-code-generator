import {
  getInputVars,
  getOutputVars
} from '../../operation-utils';

/**
 * Generate JavaScript code for a WebNN transpose operation from ONNX Transpose node info.
 * https://www.w3.org/TR/webnn/#api-mlgraphbuilder-transpose
 */
export function Transpose(
  node: any,
  toJsVarName: (name: string) => string,
  options?: { [key: string]: any } = {}
): string {
  const inputVars = getInputVars(node, toJsVarName);
  const outputVars = getOutputVars(node, toJsVarName);
  const attrs: any[] = node.attributes || [];

  // Default perm is reversed order if not specified
  let perm: number[] | null = null;
  for (const attr of attrs) {
    if (attr.name === 'perm') {
      if (attr.value.type === "int64[]" || attr.value.type === "bigint[]") {
        perm = attr.value.value.map((v: string) => Number(v));
      } else {
        perm = attr.value.value.map((v: any) => Number(v));
      }
      break;
    }
  }

  if (perm !== null) {
    return `
    const ${outputVars[0]} = builder.transpose(
      ${inputVars[0]},
      { permutation: [${perm.join(', ')}] }
    );`;
  } else {
    return `
    const ${outputVars[0]} = builder.transpose(
      ${inputVars[0]}
    );`;
  }
}