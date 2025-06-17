import {
  getInputVars,
  getOutputVars,
  getAttrValue
} from '../../operation-utils';

/**
 * Generate JavaScript code for a WebNN cast operation from ONNX Cast node info.
 * https://www.w3.org/TR/webnn/#api-mlgraphbuilder-cast
 * https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/webnn/builders/impl/cast_op_builder.cc
 */
export function Cast(
  node: any,
  toJsVarName: (name: string) => string
): string {
  const inputVars = getInputVars(node, toJsVarName);
  const outputVars = getOutputVars(node, toJsVarName);

  // Use getAttrValue for robust attribute extraction
  const toType = getAttrValue(node, 'to', 'float32');
  const label = node.name ? `{ label: '${node.name}' }` : '';

  return `
    const ${outputVars[0]} = builder.cast(
      ${inputVars[0]},
      '${toType}',
      ${label}
    );`;
}