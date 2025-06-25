import {
  getInputVars,
  getOutputVars,
  getShape,
  getAttrValue
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
  const { shape: inputShape } = getShape(node, 0, false);

  // Use getAttrValue for robust attribute extraction
  let permutation = getAttrValue(node, 'perm', undefined);

  if (!permutation) {
    // Default: reverse the axes
    permutation = [];
    for (let i = inputShape.length - 1; i >= 0; i--) {
      permutation.push(i);
    }
  }

  // Only add permutation if it's a non-empty array
  const opts: string[] = [];
  if (Array.isArray(permutation) && permutation.length > 0) {
    opts.push(`permutation: [${permutation.map(Number).join(', ')}]`);
  }
  if (node.name) opts.push(`label: '${node.name}'`);

  const optsString = opts.length ? `{ ${opts.join(', ')} }` : '';

  return `
    const ${outputVars[0]} = builder.transpose(
      ${inputVars[0]},
      ${optsString}
    );`;
}