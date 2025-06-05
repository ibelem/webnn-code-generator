import { getInputVars, getOutputVars } from '../../operation-utils';

/**
 * Generate JavaScript code for a WebNN averagePool2d operation from ONNX AveragePool node info.
 * https://www.w3.org/TR/webnn/#api-mlgraphbuilder-pool2d-average
 */
export function AveragePool(
  node: any,
  toJsVarName: (name: string) => string,
  options: { nhwc?: boolean } = {}
): string {
  const inputVars = getInputVars(node, toJsVarName);
  const outputVars = getOutputVars(node, toJsVarName);
  const attrs: any[] = node.attributes || [];
  const nhwc = !!options.nhwc;

  // Build options for builder.averagePool2d
  const poolOpts: string[] = [];
  for (const attr of attrs) {
    if (attr.name === 'kernelShape') {
      poolOpts.push(`windowDimensions: [${attr.value.value.join(', ')}]`);
    } else if (attr.name === 'pads') {
      poolOpts.push(`padding: [${attr.value.value.join(', ')}]`);
    } else if (attr.name === 'strides') {
      poolOpts.push(`strides: [${attr.value.value.join(', ')}]`);
    }
    // Add other attributes as needed
  }
  if (nhwc) {
    poolOpts.push(`layout: 'nhwc'`);
  }

  return `
const ${outputVars[0]} = builder.averagePool2d(
  ${inputVars[0]},
  {
    ${poolOpts.join(',\n    ')}
  }
);`;
}