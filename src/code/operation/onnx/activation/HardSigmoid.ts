import {
  getInputVars,
  getOutputVars,
  getAttrValue
} from '../../operation-utils';

/**
 * Generate JavaScript code for a WebNN hardSigmoid operation from ONNX HardSigmoid node info.
 * https://www.w3.org/TR/webnn/#api-mlgraphbuilder-hard-sigmoid
 * https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/webnn/builders/impl/activation_op_builder.cc
 */

export function HardSigmoid(
  node: any,
  toJsVarName: (name: string) => string
): string {
  const inputVars = getInputVars(node, toJsVarName);
  const outputVars = getOutputVars(node, toJsVarName);

  // Use getAttrValue for robust extraction with ONNX defaults: alpha=0.2, beta=0.5
  const alpha = getAttrValue(node, 'alpha', 0.2);
  const beta = getAttrValue(node, 'beta', 0.5);

  const opts = `{ alpha: ${alpha}, beta: ${beta}${node.name ? `, label: '${node.name}'` : ''} }`;

  return `
    const ${outputVars[0]} = builder.hardSigmoid(
      ${inputVars[0]},
      ${opts}
    );`;
}