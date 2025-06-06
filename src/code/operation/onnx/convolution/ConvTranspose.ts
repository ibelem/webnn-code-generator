import {
  getInputVars,
  getOutputVars,
  getShape
} from '../../operation-utils';

/**
 * Generate JavaScript code for a WebNN convTranspose2d operation from ONNX node info.
 * https://www.w3.org/TR/webnn/#api-mlgraphbuilder-convtranspose2d
 */
export function ConvTranspose(
  node: any,
  toJsVarName: (name: string) => string,
  options: { [key: string]: any } = {}
): string {
  const nhwc = !!options.nhwc;
  const inputVars = getInputVars(node, toJsVarName);
  const outputVars = getOutputVars(node, toJsVarName);

  // Get shapes for debugging
  const inputShape = getShape(node, 0, nhwc);
  const filterShape = getShape(node, 1, nhwc);

  // Debug information to confirm shapes
  let debugComment = `
    // ConvTranspose:
    // Input shape: [${inputShape}]
    // Filter shape: [${filterShape}]
    // NHWC mode: ${nhwc}`;

  let filterLayout = undefined;
  let inputLayout = undefined;
  let filterVar = inputVars[1];

  if (nhwc) {
    inputLayout = "'nhwc'";
    filterLayout = "'ohwi'";
    debugComment += `
    // Using filterLayout: ${filterLayout}, inputLayout: ${inputLayout}`;

    // Check if input channels match filter's input channels (last dim)
    if (inputShape[3] !== filterShape[3]) {
      debugComment += `
    // ERROR: input channels (${inputShape[3]}) != filter input channels (${filterShape[3]}).
    // This usually means your weights file or model export is incorrect.`;
    // Continue to generate the code, but warn the user
    }
  }

  // Attribute extraction
  const attrs: any[] = node.attributes || [];
  const attrDict: Record<string, any> = {};
  for (const attr of attrs) attrDict[attr.name] = attr;

  // Strides
  let strides = attrDict['strides']?.value?.value || [1, 1];
  if (strides.length === 1) strides = [strides[0], strides[0]];
  const strides_js = `[${strides.map((s: any) => String(Number(s))).join(', ')}]`;

  // Pads & auto_pad
  let pads = attrDict['pads']?.value?.value;
  let pads_js = '[0, 0, 0, 0]';
  let autoPad = attrDict['auto_pad']?.value;
  if (autoPad && typeof autoPad === 'string' && autoPad !== 'NOTSET') {
    pads_js = `'${autoPad}'`;
  } else if (Array.isArray(pads) && pads.length === 4) {
    const pads_webnn = [pads[0], pads[2], pads[1], pads[3]];
    pads_js = `[${pads_webnn.map((p: any) => String(Number(p))).join(', ')}]`;
  }

  // Dilations
  let dilations = attrDict['dilations']?.value?.value || [1, 1];
  if (dilations.length === 1) dilations = [dilations[0], dilations[0]];
  const dilations_js = `[${dilations.map((d: any) => String(Number(d))).join(', ')}]`;

  // Groups
  let groups = attrDict['group']?.value?.value ?? 1;
  const groups_js = String(Number(groups));

  // Output shape (optional)
  let output_shape = attrDict['output_shape']?.value?.value;
  const output_sizes_js = Array.isArray(output_shape)
    ? `[${output_shape.map((s: any) => String(Number(s))).join(', ')}]`
    : undefined;

  // Output padding (optional)
  let output_padding = attrDict['output_padding']?.value?.value;
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

  return `${debugComment}
    const ${outputVars[0]} = builder.convTranspose2d(
      ${inputVars[0]}, ${filterVar},
      {
        ${optionsArr.join(',\n        ')}
      }
    );
  `;
}