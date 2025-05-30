import { getInputVars, getOutputVars, getShape, getDtype, validateDtype, validateRank } from '../../operation-utils';

/**
 * Generate JavaScript code for WebNN matmul operation from ONNX node info.
 * https://www.w3.org/TR/webnn/#api-mlgraphbuilder-matmul
 */
export function MatMul(
  node: any,
  toJsVarName: (name: string) => string
): string {
  const inputVars = getInputVars(node, toJsVarName);
  const outputVars = getOutputVars(node, toJsVarName);
  const aShape = getShape(node, 0);
  const bShape = getShape(node, 1);
  validateDtype(getDtype(node, 0), ['float32', 'float16'], 'MatMul');
  validateDtype(getDtype(node, 1), ['float32', 'float16'], 'MatMul');
  validateRank(aShape, 2, 'MatMul');
  validateRank(bShape, 2, 'MatMul');

  return `
    const ${outputVars[0]} = builder.matmul(
      ${inputVars[0]},
      ${inputVars[1]}
    );`;
}