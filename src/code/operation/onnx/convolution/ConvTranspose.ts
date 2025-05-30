import {
  getInputVars,
  getOutputVars
} from '../../operation-utils';

/**
 * Generate JavaScript code for a WebNN convTranspose2d operation from ONNX node info.
 * https://www.w3.org/TR/webnn/#api-mlgraphbuilder-convtranspose2d
 */
function getTransposedFilterVarName(originalVar: string) {
  // Placeholder for weight transposition logic
  return `${originalVar}_transposed`;
}

export function ConvTranspose(
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
  const strides_js = Array.isArray(strides) && strides.length === 2
    ? `[${strides.map((s: any) => String(Number(s))).join(', ')}]`
    : '[1, 1]';

  // Pads
  let pads = attrDict['pads']?.value?.value;
  let pads_js = '[0, 0, 0, 0]';
  if (Array.isArray(pads) && pads.length === 4) {
    // ONNX: [top, left, bottom, right] -> WebNN: [top, bottom, left, right]
    const pads_webnn = [pads[0], pads[2], pads[1], pads[3]];
    pads_js = `[${pads_webnn.map((p: any) => String(Number(p))).join(', ')}]`;
  }

  // Dilations
  let dilations = attrDict['dilations']?.value?.value;
  const dilations_js = Array.isArray(dilations) && dilations.length === 2
    ? `[${dilations.map((d: any) => String(Number(d))).join(', ')}]`
    : '[1, 1]';

  // Groups
  let groups = attrDict['group']?.value?.value ?? 1;
  const groups_js = String(Number(groups));

  // Output shape (optional)
  let output_shape = attrDict['output_shape']?.value?.value;
  const output_sizes_js = Array.isArray(output_shape)
    ? `[${output_shape.map((s: any) => String(Number(s))).join(', ')}]`
    : undefined;

  // Bias input (optional)
  const biasVar = inputVars.length > 2 ? inputVars[2] : undefined;

  // Filter layout and transposition for NHWC
  let filterLayout = undefined;
  let inputLayout = undefined;
  let filterVarName = inputVars[1];
  if (nhwc) {
    // ONNX IOHW -> WebNN OHWI for NHWC
    filterVarName = getTransposedFilterVarName(filterVarName);
    filterLayout = "'ohwi'";
    inputLayout = "'nhwc'";
  }

  // Build options
  const optionsArr: string[] = [
    `strides: ${strides_js}`,
    `padding: ${pads_js}`,
    `dilations: ${dilations_js}`,
    `groups: ${groups_js}`
  ];
  if (output_sizes_js) optionsArr.push(`outputSizes: ${output_sizes_js}`);
  if (biasVar) optionsArr.push(`bias: ${biasVar}`);
  if (filterLayout) optionsArr.push(`filterLayout: ${filterLayout}`);
  if (inputLayout) optionsArr.push(`inputLayout: ${inputLayout}`);

  return `
    const ${outputVars[0]} = builder.convTranspose2d(
      ${inputVars[0]}, ${filterVarName},
      {
        ${optionsArr.join(',\n        ')}
      }
    );`;
}