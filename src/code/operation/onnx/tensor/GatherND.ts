import {
  getInputVars,
  getOutputVars
} from '../../operation-utils';

/**
 * Generate JavaScript code for a WebNN gatherND operation from ONNX GatherND node info.
 * https://www.w3.org/TR/webnn/#api-mlgraphbuilder-gathernd
 * https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/webnn/builders/impl/gatherND_op_builder.cc
 * Only supports batch_dims = 0 (default).
 */
export function GatherND(
  node: any,
  toJsVarName: (name: string) => string
): string {
  const inputVars = getInputVars(node, toJsVarName);
  const outputVars = getOutputVars(node, toJsVarName);

  // Only batch_dims = 0 is supported by WebNN
  // let batchDims = getAttrValue(node, 'batch_dims', 0);
  // WebNN gatherND only supports batch_dims = 0');

  const labelOpt = node.name ? `{ label: '${node.name}' }` : '';

  return `
    const ${outputVars[0]} = builder.gatherND(
      ${inputVars[0]},
      ${inputVars[1]},
      ${labelOpt}
    );`;
}