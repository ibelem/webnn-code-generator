import {
  getInputVars,
  getOutputVars,
  getAttrValue
} from '../../operation-utils';

/**
 * Generate JavaScript code for a WebNN convTranspose2d operation from ONNX node info.
 * https://www.w3.org/TR/webnn/#api-mlgraphbuilder-convtranspose2d
 * https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/webnn/builders/impl/conv_op_builder.cc
 */
export function ConvTranspose(
  node: any,
  toJsVarName: (name: string) => string,
  options: { [key: string]: any } = {}
): string {
  const nhwc = !!options.nhwc;
  const inputVars = getInputVars(node, toJsVarName);
  const outputVars = getOutputVars(node, toJsVarName);

  let filterLayout = undefined;
  let inputLayout = undefined;
  let filterVar = inputVars[1];

  if (nhwc) {
    inputLayout = "'nhwc'";
    filterLayout = "'ohwi'";
  }

  let strides = getAttrValue(node, 'strides', undefined) || [1, 1];
    if (!strides) {
    const strideH = getAttrValue(node, 'stride_h', 1);
    const strideW = getAttrValue(node, 'stride_w', 1);
    strides = [strideH, strideW];
  }
  if (strides.length === 1) strides = [strides[0], strides[0]];
  const strides_js = `[${strides.map((s: any) => String(Number(s))).join(', ')}]`;

  // Pads & auto_pad
  let pads = getAttrValue(node, 'pads', undefined);
  let pads_js = '[0, 0, 0, 0]';
    let autoPad = getAttrValue(node, 'auto_pad', undefined) || getAttrValue(node, 'padding', undefined);
  if (autoPad && typeof autoPad === 'string' && autoPad !== 'NOTSET') {
    pads_js = `'${autoPad}'`;
  } else if (Array.isArray(pads) && pads.length === 4) {
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

  // Output shape (optional)
  let output_shape = getAttrValue(node, 'output_shape', undefined);
  const output_sizes_js = Array.isArray(output_shape)
    ? `[${output_shape.map((s: any) => String(Number(s))).join(', ')}]`
    : undefined;

  // Output padding (optional)
  let output_padding = getAttrValue(node, 'output_padding', undefined);
  if (!output_padding) output_padding = [0, 0];
  if (output_padding.length === 1) output_padding.push(0);
  const output_padding_js = `[${output_padding.map((p: any) => String(Number(p))).join(', ')}]`;

  // Bias input (optional)
  const biasVar = inputVars.length > 2 ? inputVars[2] : undefined;

  // Add label for debugging if node.name exists
  const label = node.name ? `label: '${node.name}'` : undefined;

  // Build options
  const optionsArr: string[] = [
    `strides: ${strides_js}`,
    `padding: ${pads_js}`,
    `dilations: ${dilations_js}`,
    `groups: ${groups_js}`,
    `outputPadding: ${output_padding_js}`
  ];
  if (output_sizes_js) optionsArr.push(`outputSizes: ${output_sizes_js}`);
  if (biasVar) optionsArr.push(`bias: ${biasVar}`);
  if (filterLayout) optionsArr.push(`filterLayout: ${filterLayout}`);
  if (inputLayout) optionsArr.push(`inputLayout: ${inputLayout}`);
  if (label) optionsArr.push(label);

  return `
    const ${outputVars[0]} = builder.convTranspose2d(
      ${inputVars[0]}, ${filterVar},
      {
        ${optionsArr.join(',\n        ')}
      }
    );
  `;
}