import {
  getInputVars,
  getOutputVars
} from '../operation-utils';

/**
 * Generate JavaScript code for a WebNN einsum operation from ONNX Einsum node info.
 * https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/webnn/builders/impl/einsum_op_builder.cc
 * Note: WebNN does NOT have a native einsum operator. The closest is matmul, but einsum is more general.
 * This implementation only supports the common matmul-like cases (e.g. "ij,jk->ik").
 * For more complex einsum equations, you would need to implement additional logic or throw an error.
 */
export function Einsum(
  node: any,
  toJsVarName: (name: string) => string
): string {
  const inputVars = getInputVars(node, toJsVarName);
  const outputVars = getOutputVars(node, toJsVarName);

  // Extract the equation string from ONNX node attributes
  let equation = '';
  for (const attr of node.attributes || []) {
    if (attr.name === 'equation') {
      equation = typeof attr.value === 'string' ? attr.value : attr.value?.value;
      break;
    }
  }

  // Only support matmul-like einsum equations for direct mapping
  // e.g. "ij,jk->ik" or "ab,bc->ac"
  const matmulPatterns = [
    /^([a-z]{2}),([a-z]{2})->([a-z]{2})$/i, // e.g. "ij,jk->ik"
    /^([a-z]{2}),([a-z]{2})$/i              // e.g. "ij,jk"
  ];
  const isMatmulLike = matmulPatterns.some(re => re.test(equation.replace(/\s/g, '')));

  if (isMatmulLike && inputVars.length === 2) {
    // Use WebNN matmul for supported einsum patterns, add label
    const labelOpt = node.name ? `{ label: '${node.name}_matmul' }` : '{}';
    return `
      const ${outputVars[0]} = builder.matmul(
        ${inputVars[0]},
        ${inputVars[1]},
        ${labelOpt}
      );`;
  } else {
    // For general einsum, not supported by WebNN directly
    return `
      throw new Error('Einsum equation "${equation}" is not supported by WebNN codegen.');
    `;
  }
}