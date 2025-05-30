import {
  getInputVars,
  getOutputVars,
  getShape
} from '../../operation-utils';

/**
 * Generate JavaScript code for a WebNN conv2d operation from ONNX node info.
 * https://www.w3.org/TR/webnn/#api-mlgraphbuilder-conv2d
 */
export function Conv(
  node: any,
  toJsVarName: (name: string) => string,
  options: { [key: string]: any } = {}
): string {
  const nhwc = !!options.nhwc;
  const inputVars = getInputVars(node, toJsVarName);
  const outputVars = getOutputVars(node, toJsVarName);

  // Attribute extraction
  const attrs: any[] = node.attributes || [];
  const attrDict: Record<string, any> = {};
  for (const attr of attrs) attrDict[attr.name] = attr;

  // Strides
  let strides = attrDict['strides']?.value?.value;
  if (!strides && (attrDict['stride_w'] || attrDict['stride_h'])) {
    const stride_w = Number(attrDict['stride_w']?.value ?? 1);
    const stride_h = Number(attrDict['stride_h']?.value ?? 1);
    strides = [stride_h, stride_w];
  }
  const strides_js = Array.isArray(strides) && strides.length === 2
    ? `[${strides.map((s: any) => String(Number(s))).join(', ')}]`
    : '[1, 1]';

  // Pads
  let pads = attrDict['pads']?.value?.value;
  let pads_js = '[0, 0, 0, 0]';
  if (!pads && attrDict['padding']?.value) {
    const padType = attrDict['padding'].value;
    pads_js = padType === 'VALID' ? '[0, 0, 0, 0]' : `'${padType}'`;
  } else if (Array.isArray(pads) && pads.length === 4) {
    // ONNX: [top, left, bottom, right] -> WebNN: [top, bottom, left, right]
    const pads_webnn = [pads[0], pads[2], pads[1], pads[3]];
    pads_js = `[${pads_webnn.map((p: any) => String(Number(p))).join(', ')}]`;
  }

  // Dilations
  let dilations = attrDict['dilations']?.value?.value;
  if (!dilations && (attrDict['dilation_w_factor'] || attrDict['dilation_h_factor'])) {
    const dilation_w = Number(attrDict['dilation_w_factor']?.value ?? 1);
    const dilation_h = Number(attrDict['dilation_h_factor']?.value ?? 1);
    dilations = [dilation_h, dilation_w];
  }
  const dilations_js = Array.isArray(dilations) && dilations.length === 2
    ? `[${dilations.map((d: any) => String(Number(d))).join(', ')}]`
    : '[1, 1]';

  // Groups
  let groups = attrDict['group']?.value?.value ?? 1;
  const groups_js = String(Number(groups));

  // Bias input (optional)
  const biasVar = inputVars.length > 2 ? inputVars[2] : undefined;

  // Filter shape and layout
  const filterShape = getShape(node, 1);
  let inputLayout = nhwc ? "'nhwc'" : "'nchw'";
  let filterLayout = "'oihw'";
  let filterVarName = inputVars[1];

  // Depthwise detection
  let isDepthwise = false;
  if (groups !== 1 && filterShape.length === 4) {
    const outputChannels = filterShape[0];
    if (groups === outputChannels) isDepthwise = true;
  }

  // Helper: transpose filter weights for NHWC (placeholder)
  function getTransposedFilterVarName(originalVar: string, _permutation: number[]) {
    return `${originalVar}_transposed`;
  }

  if (nhwc) {
    if (isDepthwise) {
      // Depthwise: OIHW -> IHWO
      filterVarName = getTransposedFilterVarName(filterVarName, [1, 2, 3, 0]);
      filterLayout = "'ihwo'";
    } else {
      // Regular: OIHW -> OHWI
      filterVarName = getTransposedFilterVarName(filterVarName, [0, 2, 3, 1]);
      filterLayout = "'ohwi'";
    }
    inputLayout = "'nhwc'";
  }

  // Build options
  const optionsArr: string[] = [
    `strides: ${strides_js}`,
    `padding: ${pads_js}`,
    `dilations: ${dilations_js}`,
    `groups: ${groups_js}`
  ];
  if (biasVar) optionsArr.push(`bias: ${biasVar}`);
  if (filterLayout) optionsArr.push(`filterLayout: ${filterLayout}`);
  if (inputLayout) optionsArr.push(`inputLayout: ${inputLayout}`);

  return `
    const ${outputVars[0]} = builder.conv2d(
      ${inputVars[0]}, ${filterVarName},
      {
        ${optionsArr.join(',\n        ')}
      }
    );`;
}