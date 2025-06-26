import {
  getInputVars,
  getOutputVars,
  getAttrValue
} from '../../operation-utils';

/**
 * Generate JavaScript code for a WebNN maxPool2d operation from ONNX MaxPool node info.
 */
export function MaxPool(
  node: any,
  toJsVarName: (name: string) => string,
  options: { [key: string]: any } = {}
): string {
  const nhwc = !!options.nhwc;
  const inputVars = getInputVars(node, toJsVarName);
  const outputVars = getOutputVars(node, toJsVarName);

  // Use getAttrValue for robust extraction
  const kernelShape = getAttrValue(node, 'kernel_shape', [0, 0]);
  const strides = getAttrValue(node, 'strides', [1, 1]);
  const pads = getAttrValue(node, 'pads', [0, 0, 0, 0]);
  const dilations = getAttrValue(node, 'dilations', [1, 1]);
  const ceilMode = getAttrValue(node, 'ceil_mode', 0);

  // WebNN expects [beginH, endH, beginW, endW], ONNX is [beginH, beginW, endH, endW]
  let paddingOpt = '';
  if (pads && pads.length === 4) {
    paddingOpt = `padding: [${pads[0]}, ${pads[2]}, ${pads[1]}, ${pads[3]}]`;
  }

  // Build options
  const opts: string[] = [];
  if (kernelShape) opts.push(`windowDimensions: [${kernelShape.join(', ')}]`);
  if (paddingOpt) opts.push(paddingOpt);
  if (strides) opts.push(`strides: [${strides.join(', ')}]`);
  if (dilations) opts.push(`dilations: [${dilations.join(', ')}]`);
  opts.push(`layout: '${nhwc ? 'nhwc' : 'nchw'}'`);
  opts.push(`roundingType: '${ceilMode ? 'ceil' : 'floor'}'`);
  if (node.name) opts.push(`label: '${node.name}'`);

  return `
    const ${outputVars[0]} = builder.maxPool2d(
      ${inputVars[0]},
      {
        ${opts.join(',\n        ')}
      }
    );`;
}