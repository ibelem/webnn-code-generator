import {
  getInputVars,
  getOutputVars,
  getShape,
  getAttrValue
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

  // Strides: ONNX uses 'strides', TFLite may use 'stride_h' and 'stride_w'
  let strides = getAttrValue(node, 'strides', undefined);
  if (!strides) {
    const strideH = getAttrValue(node, 'stride_h', 1);
    const strideW = getAttrValue(node, 'stride_w', 1);
    strides = [strideH, strideW];
  }

  if (!strides) strides = [1, 1];
  if (strides.length === 1) strides = [strides[0], strides[0]];
  const strides_js = `[${strides.map((s: any) => String(Number(s))).join(', ')}]`;

  // Pads & auto_pad
  let pads = getAttrValue(node, 'pads', undefined);
  let pads_js = '[0, 0, 0, 0]';
  let autoPad = getAttrValue(node, 'auto_pad', undefined) || getAttrValue(node, 'padding', undefined);
  if (autoPad && typeof autoPad === 'string' && autoPad !== 'NOTSET') {
    // Pass autoPad string directly if present
    pads_js = `'${autoPad}'`;
  } else if (Array.isArray(pads) && pads.length === 4) {
    // ONNX: [top, left, bottom, right] -> WebNN: [top, bottom, left, right]
    const pads_webnn = [pads[0], pads[2], pads[1], pads[3]];
    pads_js = `[${pads_webnn.map((p: any) => String(Number(p))).join(', ')}]`;
  }

    // Dilations: ONNX uses 'dilations', TFLite may use 'dilation_h_factor' and 'dilation_w_factor'
  let dilations = getAttrValue(node, 'dilations', undefined);
  if (!dilations) {
    const dilationH = getAttrValue(node, 'dilation_h_factor', 1);
    const dilationW = getAttrValue(node, 'dilation_w_factor', 1);
    dilations = [dilationH, dilationW];
  }

  if (!dilations) dilations = [1, 1];
  if (dilations.length === 1) dilations = [dilations[0], dilations[0]];
  const dilations_js = `[${dilations.map((d: any) => String(Number(d))).join(', ')}]`;

  // Groups
  let groups = getAttrValue(node, 'group', 1);
  const groups_js = String(Number(groups));

  // Bias input (optional)
  const biasVar = inputVars.length > 2 ? inputVars[2] : undefined;

  // Filter shape and layout
  const { shape: inputShape } = getShape(node, 0, nhwc);
  const { shape: filterShape } = getShape(node, 1, nhwc);

  // Depthwise detection (NHWC: groups === inputChannels)
  let isDepthwise = false;
  if (groups !== 1 && filterShape?.length === 4) {
    if (inputShape && (nhwc ? inputShape.length > 3 : inputShape.length > 1)) {
      const inputChannels = nhwc ? inputShape[3] : inputShape[1];
      if (inputChannels && groups === inputChannels) isDepthwise = true;
    }
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
    );`;
}