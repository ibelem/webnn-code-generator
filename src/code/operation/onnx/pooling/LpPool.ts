import {
  getInputVars,
  getOutputVars,
  getAttrValue
} from '../../operation-utils';

/**
 * Generate JavaScript code for a WebNN lpPool2d operation from ONNX LpPool node info.
 */
export function LpPool(
  node: any,
  toJsVarName: (name: string) => string,
  options: { [key: string]: any } = {}
): string {
  const nhwc = !!options.nhwc;
  const inputVars = getInputVars(node, toJsVarName);
  const outputVars = getOutputVars(node, toJsVarName);
  const attrs: any[] = node.attributes || [];

  // Use getAttrValue for robust extraction
  const kernelShape = getAttrValue(attrs, 'kernel_shape', [0, 0]);
  const strides = getAttrValue(attrs, 'strides', [1, 1]);
  const pads = getAttrValue(attrs, 'pads', [0, 0, 0, 0]);
  const p = getAttrValue(attrs, 'p', 2); // Default to 2 for L2 norm
  const ceilMode = getAttrValue(attrs, 'ceil_mode', 0);

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
  opts.push(`normType: ${p}`);
  opts.push(`layout: '${nhwc ? 'nhwc' : 'nchw'}'`);
  opts.push(`roundingType: '${ceilMode ? 'ceil' : 'floor'}'`);
  if (node.name) opts.push(`label: '${node.name}'`);

  return `
    const ${outputVars[0]} = builder.lpPool2d(
      ${inputVars[0]},
      {
        ${opts.join(',\n    ')}
      }
    );`;
}