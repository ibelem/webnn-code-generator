/**
 * Generate JavaScript code for a WebNN conv2d operation from ONNX node info.
 * @param node - The ONNX node object (with inputs, outputs, attributes)
 * @param toJsVarName - Function to convert ONNX names to JS variable names
 * @returns JavaScript code string for the conv2d operation
 */

/**
 * WebNN Specification: https://www.w3.org/TR/webnn/
 * https://www.w3.org/TR/webnn/#api-mlgraphbuilder-conv2d
 */

import { getNonEmptyStringAroundNewline } from '../../utils';

// Helper: transpose filter weights for NHWC (not implemented here, but you should handle this in your weight loader)
function getTransposedFilterVarName(originalVar: string, permutation: number[]) {
  // This is a placeholder. In practice, you should transpose the weights in your loader and return the new var name.
  // For codegen, just append '_transposed' for clarity.
  console.log(`Transposing filter weights for NHWC: ${originalVar} with permutation ${permutation}`);
  return `${originalVar}_transposed`;
}

export function conv2d_js(
  node: any,
  toJsVarName: (name: string) => string,
  nhwc: boolean = false // Pass true if you want NHWC layout
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
  if (!strides && (attrDict['stride_w'] || attrDict['stride_h'])) {
    const stride_w = Number(attrDict['stride_w']?.value ?? 1);
    const stride_h = Number(attrDict['stride_h']?.value ?? 1);
    strides = [stride_h, stride_w];
  }
  if (Array.isArray(strides) && strides.length === 2) {
    strides_js = `[${strides.map((s: any) => String(Number(s))).join(', ')}]`;
  }

  // Pads
  let pads_js = '[0, 0, 0, 0]';
  let pads = attrDict['pads']?.value?.value;
  if (!pads && attrDict['padding']?.value) {
    const padType = attrDict['padding'].value;
    if (padType === 'VALID') {
      pads_js = '[0, 0, 0, 0]';
    } else if (padType === 'SAME') {
      pads_js = `'${padType}'`;
    }
  } else if (Array.isArray(pads) && pads.length === 4) {
    // ONNX: [top, left, bottom, right] -> WebNN: [top, bottom, left, right]
    const pads_webnn = [pads[0], pads[2], pads[1], pads[3]];
    pads_js = `[${pads_webnn.map((p: any) => String(Number(p))).join(', ')}]`;
  }

  // Dilations
  let dilations_js = '[1, 1]';
  let dilations = attrDict['dilations']?.value?.value;
  if (!dilations && (attrDict['dilation_w_factor'] || attrDict['dilation_h_factor'])) {
    const dilation_w = Number(attrDict['dilation_w_factor']?.value ?? 1);
    const dilation_h = Number(attrDict['dilation_h_factor']?.value ?? 1);
    dilations = [dilation_h, dilation_w];
  }
  if (Array.isArray(dilations) && dilations.length === 2) {
    dilations_js = `[${dilations.map((d: any) => String(Number(d))).join(', ')}]`;
  }

  // Groups
  let groups = attrDict['group']?.value?.value ?? 1;
  let groups_js = String(Number(groups));

  // Bias input (optional)
  const biasVar = inputVars.length > 2 ? inputVars[2] : undefined;

  // Filter layout and transposition for NHWC
  let filterLayout = undefined;
  let inputLayout = undefined;
  let filterVarName = inputVars[1];

  // Try to detect depthwise conv
  let isDepthwise = false;
  let filterShape = attrDict['kernel_shape']?.value || []; // fallback: you may need to get this from weights meta
  if (groups !== 1 && filterShape.length === 4) {
    const outputChannels = filterShape[0];
    if (groups === outputChannels) isDepthwise = true;
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
  const options: string[] = [
    `strides: ${strides_js}`,
    `padding: ${pads_js}`,
    `dilations: ${dilations_js}`,
    `groups: ${groups_js}`
  ];

  if(biasVar) options.push(`bias: ${biasVar}`);
  if (filterLayout) options.push(`filterLayout: ${filterLayout}`);
  if (inputLayout) options.push(`inputLayout: ${inputLayout}`);

  return `
    const ${outputVar} = builder.conv2d(
      ${inputVars[0]}, ${filterVarName},
      {
        ${options.join(',\n        ')}
      }
    );`;
}