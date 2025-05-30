import {
  getInputVars,
  getOutputVars
} from '../../operation-utils';

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

  return `
    const ${outputVars[0]} = builder.matmul(
      ${inputVars[0]},
      ${inputVars[1]}
    );`;
}