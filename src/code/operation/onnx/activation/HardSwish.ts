import {
  getInputVars,
  getOutputVars
} from '../../operation-utils';

/**
 * Generate JavaScript code for a WebNN hardSwish operation from ONNX HardSwish node info.
 * https://www.w3.org/TR/webnn/#api-mlgraphbuilder-hard-swish
 */
export function HardSwish(
  node: any,
  toJsVarName: (name: string) => string
): string {
  const inputVars = getInputVars(node, toJsVarName);
  const outputVars = getOutputVars(node, toJsVarName);

  return `
    const ${outputVars[0]} = builder.hardSwish(
      ${inputVars[0]}
    );`;
}