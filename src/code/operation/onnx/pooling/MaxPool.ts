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

  // Layout
  const nhwc = !!options.nhwc;
  const layout = nhwc ? "'nhwc'" : "'nchw'";

  // Build options
  const opts: string[] = [];
  if (kernelShape) opts.push(`windowDimensions: [${kernelShape.join(', ')}]`);
  if (pads) opts.push(`padding: [${pads.join(', ')}]`);
  if (strides) opts.push(`strides: [${strides.join(', ')}]`);
  if (dilations) opts.push(`dilations: [${dilations.join(', ')}]`);
  opts.push(`layout: ${layout}`);
  opts.push(`roundingType: '${ceilMode ? 'ceil' : 'floor'}'`);

  return `
    const ${outputVars[0]} = builder.maxPool2d(
      ${inputVars[0]},
      {
        ${opts.join(',\n        ')}
      }
    );`;
}