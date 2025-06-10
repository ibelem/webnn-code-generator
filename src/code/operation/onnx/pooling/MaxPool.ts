import {
  getInputVars,
  getOutputVars
} from '../../operation-utils';

/**
 * Generate JavaScript code for a WebNN maxPool2d operation from ONNX MaxPool node info.
 * https://www.w3.org/TR/webnn/#api-mlgraphbuilder-pool2d-max
 */
export function MaxPool(
  node: any,
  toJsVarName: (name: string) => string,
  options: { [key: string]: any } = {}
): string {
  const inputVars = getInputVars(node, toJsVarName);
  const outputVars = getOutputVars(node, toJsVarName);
  const attrs: any[] = node.attributes || [];
  const attrDict: Record<string, any> = {};
  for (const attr of attrs) attrDict[attr.name] = attr;

  // Extract ONNX attributes
  const kernelShape = attrDict['kernel_shape']?.value;
  const strides = attrDict['strides']?.value;
  const pads = attrDict['pads']?.value;
  const dilations = attrDict['dilations']?.value;
  const ceilMode = attrDict['ceil_mode']?.value ?? 0;
  const nhwc = !!options.nhwc;

  // WebNN expects [beginH, endH, beginW, endW], ONNX is [beginH, beginW, endH, endW]
  let paddingStr = '';
  if (pads && pads.length === 4) {
    paddingStr = `padding: [${pads[0]}, ${pads[2]}, ${pads[1]}, ${pads[3]}]`;
  }

  // Build options
  const opts: string[] = [];
  if (kernelShape) opts.push(`windowDimensions: [${kernelShape.join(', ')}]`);
  if (paddingStr) opts.push(paddingStr);
  if (strides) opts.push(`strides: [${strides.join(', ')}]`);
  if (dilations) opts.push(`dilations: [${dilations.join(', ')}]`);
  opts.push(`layout: '${nhwc ? 'nhwc' : 'nchw'}'`);
  opts.push(`roundingType: '${ceilMode ? 'ceil' : 'floor'}'`);
  if (node.name) opts.push(`label: '${node.name}'`);

  return `
    const ${outputVars[0]} = builder.maxPool2d(
      ${inputVars[0]},
      {
        ${opts.join(',\n    ')}
      }
    );`;
}