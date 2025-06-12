import {
  getInputVars,
  getOutputVars
} from '../../operation-utils';

/**
 * Generate JavaScript code for a WebNN averagePool2d operation from ONNX GlobalAveragePool node info.
 * https://www.w3.org/TR/webnn/#api-mlgraphbuilder-pool2d-average
 * https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/webnn/builders/impl/pool_op_builder.cc
 */
export function GlobalAveragePool(
  node: any,
  toJsVarName: (name: string) => string,
  options: { nhwc?: boolean } = {}
): string {
  const inputVars = getInputVars(node, toJsVarName);
  const outputVars = getOutputVars(node, toJsVarName);
  const nhwc = !!options.nhwc;

  // Always set layout and label for debugging
  const opts: string[] = [
    `layout: '${nhwc ? 'nhwc' : 'nchw'}'`
  ];
  if (node.name) {
    opts.push(`label: '${node.name}'`);
  }

  return `
    const ${outputVars[0]} = builder.averagePool2d(
      ${inputVars[0]},
      { ${opts.join(', ')} }
    );`;
}