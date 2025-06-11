import {
  getInputVars,
  getOutputVars
} from '../../operation-utils';

/**
 * Generate JavaScript code for a WebNN scatterND operation from ONNX ScatterND node info.
 * https://www.w3.org/TR/webnn/#api-mlgraphbuilder-scatternd
 * https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/webnn/builders/impl/scatterND_op_builder.cc
 * Only supports reduction='none' (default).
 */
export function ScatterND(
  node: any,
  toJsVarName: (name: string) => string
): string {
  const inputVars = getInputVars(node, toJsVarName); // [data, indices, updates]
  const outputVars = getOutputVars(node, toJsVarName);

  // Only reduction='none' is supported
  for (const attr of node.attributes || []) {
    if (attr.name === 'reduction' && attr.value !== 'none') {
      throw new Error('WebNN scatterND only supports reduction type "none" (default).');
    }
  }

  const labelOpt = node.name ? `{ label: '${node.name}' }` : `{}`;

  return `
    const ${outputVars[0]} = builder.scatterND(
      ${inputVars[0]},
      ${inputVars[1]},
      ${inputVars[2]},
      ${labelOpt}
    );
`;
}