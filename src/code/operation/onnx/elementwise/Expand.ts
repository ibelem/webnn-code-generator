import {
  getInputVars,
  getOutputVars
} from '../../operation-utils';

/**
 * Generate JavaScript code for a WebNN expand operation from ONNX Expand node info.
 * https://www.w3.org/TR/webnn/#api-mlgraphbuilder-expand
 * https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/webnn/builders/impl/expand_op_builder.cc
 * The new shape must be a constant initializer.
 */
export function Expand(
  node: any,
  toJsVarName: (name: string) => string
): string {
  const inputVars = getInputVars(node, toJsVarName);
  const outputVars = getOutputVars(node, toJsVarName);

  // The new shape is expected as a constant initializer (second input)
  let newShape: number[] = [];
  if (node.inputs?.[1]?.value?.[0]?.initializer?.values) {
    // Extract shape from initializer
    const valuesObj = node.inputs[1].value[0].initializer.values;
    // ONNX stores as { "0": dim0, "1": dim1, ... }
    newShape = Object.keys(valuesObj)
      .sort((a, b) => Number(a) - Number(b))
      .map(k => Number(valuesObj[k]));
  }

  // WebNN does not support 0-dim in expand shape
  if (newShape.some(dim => dim === 0)) {
    throw new Error('WebNN expand does not support new shape with 0 dimension.');
  }

  const labelOpt = node.name ? `{ label: '${node.name}' }` : '';

  return `
    const ${outputVars[0]} = builder.expand(
      ${inputVars[0]},
      [${newShape.join(', ')}],
      ${labelOpt}
    );
`;
}