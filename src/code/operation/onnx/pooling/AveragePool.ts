import { getInputVars, getOutputVars } from '../../operation-utils';

/**
 * Generate JavaScript code for a WebNN averagePool2d operation from ONNX AveragePool node info.
 * https://www.w3.org/TR/webnn/#api-mlgraphbuilder-pool2d-average
 * https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/webnn/builders/impl/pool_op_builder.cc
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
      // WebNN expects [beginH, endH, beginW, endW], ONNX is [beginH, beginW, endH, endW]
      const pads = attr.value.value;
      if (pads.length === 4) {
        poolOpts.push(`padding: [${pads[0]}, ${pads[2]}, ${pads[1]}, ${pads[3]}]`);
      }
    } else if (attr.name === 'strides') {
      poolOpts.push(`strides: [${attr.value.value.join(', ')}]`);
    } else if (attr.name === 'dilations') {
      poolOpts.push(`dilations: [${attr.value.value.join(', ')}]`);
    } else if (attr.name === 'ceilMode') {
      // ceilMode: 0=floor, 1=ceil
      poolOpts.push(`roundingType: '${attr.value.value === 1 ? 'ceil' : 'floor'}'`);
    }
    // Add other attributes as needed
  }

  poolOpts.push(`layout: '${nhwc ? 'nhwc' : 'nchw'}'`);
  if (node.name) {
    poolOpts.push(`label: '${node.name}'`);
  }

  return `
    const ${outputVars[0]} = builder.averagePool2d(
      ${inputVars[0]},
      {
        ${poolOpts.join(',\n    ')}
      }
    );`;
}