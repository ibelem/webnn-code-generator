import {
  getInputVars,
  getOutputVars,
  getShape
} from '../../operation-utils';

/**
 * Generate JavaScript code for a WebNN conv2d operation from ONNX node info.
 * https://www.w3.org/TR/webnn/#api-mlgraphbuilder-conv2d
 * https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/webnn/builders/impl/conv_op_builder.cc
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
  if (!strides) strides = [1, 1];
  if (strides.length === 1) strides = [strides[0], strides[0]];
  const strides_js = `[${strides.map((s: any) => String(Number(s))).join(', ')}]`;

  // Pads & auto_pad
  let pads = attrDict['pads']?.value?.value;
  let pads_js = '[0, 0, 0, 0]';
  let autoPad = attrDict['auto_pad']?.value || attrDict['padding']?.value;
  if (autoPad && typeof autoPad === 'string' && autoPad !== 'NOTSET') {
    // Pass autoPad string directly if present
    pads_js = `'${autoPad}'`;
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
  if (!dilations) dilations = [1, 1];
  if (dilations.length === 1) dilations = [dilations[0], dilations[0]];
  const dilations_js = `[${dilations.map((d: any) => String(Number(d))).join(', ')}]`;

  // Groups
  let groups = attrDict['group']?.value?.value ?? 1;
  const groups_js = String(Number(groups));

  // Bias input (optional)
  const biasVar = inputVars.length > 2 ? inputVars[2] : undefined;

  // Filter shape and layout
  const filterShape = getShape(node, 1, nhwc);  // Add nhwc parameter

  // Depthwise detection (NHWC: groups === inputChannels)
  let isDepthwise = false;
  if (groups !== 1 && filterShape.length === 4) {
    const inputShape = getShape(node, 0, nhwc);  // Add nhwc parameter
    const inputChannels = nhwc ? inputShape[3] : inputShape[1];
    if (groups === inputChannels) isDepthwise = true;
  }

  let filterLayout = "'oihw'";
  let inputLayout = "'nchw'";
  if (nhwc) {
    inputLayout = "'nhwc'";
    filterLayout = isDepthwise ? "'ihwo'" : "'ohwi'";
  }

  // Add label for debugging if node.name exists
  const label = node.name ? `label: '${node.name}'` : undefined;

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
  if (label) optionsArr.push(label);

  return `
    const ${outputVars[0]} = builder.conv2d(
      ${inputVars[0]}, ${inputVars[1]},
      {
        ${optionsArr.join(',\n        ')}
      }
    );
  `;
}