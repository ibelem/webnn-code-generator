import {
  getInputVars,
  getOutputVars,
  getShape,
  getAttrValue
} from '../../operation-utils';

/**
 * Generate JavaScript code for WebNN from ONNX ConvInteger node.
 * https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/webnn/builders/impl/conv_op_builder.cc
 * Implements ConvInteger as: dequantizeLinear(x) + dequantizeLinear(w) -> conv2d -> cast(int32).
 */
export function ConvInteger(
  node: any,
  toJsVarName: (name: string) => string,
  options: { [key: string]: any } = {}
): string {
  const nhwc = !!options.nhwc;
  const inputVars = getInputVars(node, toJsVarName);
  const outputVars = getOutputVars(node, toJsVarName);

  // x, w, x_zero_point, w_zero_point
  const xVar = inputVars[0];
  const wVar = inputVars[1];
  const xZeroPoint = inputVars[2] || '0';
  const wZeroPoint = inputVars[3] || '0';

  // Dequantize x and w to float32 (scale is always 1.0 for ConvInteger)
  const xDequant = `${outputVars[0]}_x_dequant`;
  const wDequant = `${outputVars[0]}_w_dequant`;

  // Conv2d options (reuse Conv logic for strides, pads, dilations, groups, layouts)
  // You may want to refactor this if you want to share code with Conv
  // For now, copy the logic from Conv.ts

  // Strides
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
    pads_js = `'${autoPad}'`;
  } else if (Array.isArray(pads) && pads.length === 4) {
    // ONNX: [top, left, bottom, right] -> WebNN: [top, bottom, left, right]
    const pads_webnn = [pads[0], pads[2], pads[1], pads[3]];
    pads_js = `[${pads_webnn.map((p: any) => String(Number(p))).join(', ')}]`;
  }

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

  // Filter shape and layout
  const { shape: inputShape } = getShape(node, 0, nhwc);
  const { shape: filterShape } = getShape(node, 1, nhwc);

  let isDepthwise = false;
  if (groups !== 1 && filterShape.length === 4) {
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
  const label = node.name ? `label: '${node.name}_conv_dequantized_inputs'` : undefined;

  // Build options
  const optionsArr: string[] = [
    `strides: ${strides_js}`,
    `padding: ${pads_js}`,
    `dilations: ${dilations_js}`,
    `groups: ${groups_js}`
  ];
  if (filterLayout) optionsArr.push(`filterLayout: ${filterLayout}`);
  if (inputLayout) optionsArr.push(`inputLayout: ${inputLayout}`);
  if (label) optionsArr.push(label);

  return `
    const ${xDequant} = builder.dequantizeLinear(
      ${xVar},
      1.0,
      ${xZeroPoint},
      { label: '${node.name || ''}_dequantized_x' }
    );
    const ${wDequant} = builder.dequantizeLinear(
      ${wVar},
      1.0,
      ${wZeroPoint},
      { label: '${node.name || ''}_dequantized_w' }
    );
    const ${outputVars[0]}_float = builder.conv2d(
      ${xDequant}, ${wDequant},
      {
        ${optionsArr.join(',\n    ')}
      }
    );
    const ${outputVars[0]} = builder.cast(
      ${outputVars[0]}_float,
      'int32',
      { label: '${node.name || ''}_cast_output' }
    );
`;
}