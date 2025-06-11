import {
  getInputVars,
  getOutputVars,
  getShape
} from '../../operation-utils';

/**
 * Generate JavaScript code for a WebNN transpose operation from ONNX Transpose node info.
 * https://www.w3.org/TR/webnn/#api-mlgraphbuilder-transpose
 * https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/webnn/builders/impl/transpose_op_builder.cc
 */
export function Transpose(
  node: any,
  toJsVarName: (name: string) => string
): string {
  const inputVars = getInputVars(node, toJsVarName);
  const outputVars = getOutputVars(node, toJsVarName);
  const attrs: any[] = node.attributes || [];

  // Get input shape to determine default permutation if needed
  const inputShape = getShape(node, 0);

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
  if (!perm) {
    // Default: reverse the axes
    perm = [];
    for (let i = inputShape.length - 1; i >= 0; i--) {
      perm.push(i);
    }
  }

  // Add label option
  const labelOpt = node.name ? `label: '${node.name}'` : '';

  return `
  const ${outputVars[0]} = builder.transpose(
    ${inputVars[0]},
    { ${labelOpt}, permutation: [${perm.join(', ')}] }
  );`;
}