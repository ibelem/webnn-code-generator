/**
 * Generate JavaScript code for a WebNN convTranspose2d operation from ONNX node info.
 * @param node - The ONNX node object (with inputs, outputs, attributes)
 * @param toJsVarName - Function to convert ONNX names to JS variable names
 * @param nhwc - Whether to use NHWC layout (default: false)
 * @returns JavaScript code string for the convTranspose2d operation
 */

/**
 * WebNN Specification: https://www.w3.org/TR/webnn/
 * https://www.w3.org/TR/webnn/#api-mlgraphbuilder-convtranspose2d
 */

import { getNonEmptyStringAroundNewline } from '../../../../utils';

// Helper: transpose filter weights for NHWC (not implemented here, but you should handle this in your weight loader)
function getTransposedFilterVarName(originalVar: string, permutation: number[]) {
  // This is a placeholder. In practice, you should transpose the weights in your loader and return the new var name.
  // For codegen, just append '_transposed' for clarity.
  console.log(`Transposing filter weights for NHWC: ${originalVar} with permutation ${permutation}`);
  return `${originalVar}_transposed`;
}

export function ConvTranspose(
  node: any,
  toJsVarName: (name: string) => string,
  nhwc: boolean = false
): string {
  const inputs: string[] = node.inputs?.map((i: any) => getNonEmptyStringAroundNewline(i.value?.[0].name)) || [];
  const outputs: string[] = node.outputs?.map((o: any) => getNonEmptyStringAroundNewline(o.value?.[0].name)) || [];
  const attrs: any[] = node.attributes || [];
  const attrDict: Record<string, any> = {};
  for (const attr of attrs) attrDict[attr.name] = attr;

  const inputVars = inputs.map(toJsVarName);
  const outputVar = toJsVarName(outputs[0]);

  // Strides
  let strides_js = '[1, 1]';
  let strides = attrDict['strides']?.value?.value;
  if (Array.isArray(strides) && strides.length === 2) {
    strides_js = `[${strides.map((s: any) => String(Number(s))).join(', ')}]`;
  }

  // Pads
  let pads_js = '[0, 0, 0, 0]';
  let pads = attrDict['pads']?.value?.value;
  if (Array.isArray(pads) && pads.length === 4) {
    // ONNX: [top, left, bottom, right] -> WebNN: [top, bottom, left, right]
    const pads_webnn = [pads[0], pads[2], pads[1], pads[3]];
    pads_js = `[${pads_webnn.map((p: any) => String(Number(p))).join(', ')}]`;
  }

  // Dilations
  let dilations_js = '[1, 1]';
  let dilations = attrDict['dilations']?.value?.value;
  if (Array.isArray(dilations) && dilations.length === 2) {
    dilations_js = `[${dilations.map((d: any) => String(Number(d))).join(', ')}]`;
  }

  // Groups
  let groups = attrDict['group']?.value?.value ?? 1;
  let groups_js = String(Number(groups));

  // Output shape (optional)
  let output_shape = attrDict['output_shape']?.value?.value;
  let output_sizes_js = Array.isArray(output_shape)
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
    filterVarName = getTransposedFilterVarName(filterVarName, [1, 2, 3, 0]);
    filterLayout = "'ohwi'";
    inputLayout = "'nhwc'";
  }

  // Build options
  const options: string[] = [
    `strides: ${strides_js}`,
    `padding: ${pads_js}`,
    `dilations: ${dilations_js}`,
    `groups: ${groups_js}`
  ];

  if(output_sizes_js) options.push(`outputSizes: ${output_sizes_js}`);
  if(biasVar) options.push(`bias: ${biasVar}`);
  if (filterLayout) options.push(`filterLayout: ${filterLayout}`);
  if (inputLayout) options.push(`inputLayout: ${inputLayout}`);

  return `
    const ${outputVar} = builder.convTranspose2d(
      ${inputVars[0]}, ${filterVarName},
      {
        ${options.join(',\n        ')}
      }
    );`;
}