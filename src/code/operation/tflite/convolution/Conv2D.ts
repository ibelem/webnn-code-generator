/**
 * Generate JavaScript code for a WebNN conv2d operation from TFLite node info.
 * WebNN Specification: https://www.w3.org/TR/webnn/
 * https://www.w3.org/TR/webnn/#api-mlgraphbuilder-conv2d
 */

import {
  getInputVars,
  getOutputVars,
  getShape,
  getAttrValue
} from '../../operation-utils';

// Helper: transpose filter weights for NHWC (not implemented here, but you should handle this in your weight loader)
function getTransposedFilterVarName(originalVar: string, _permutation: number[]) {
  // This is a placeholder. In practice, you should transpose the weights in your loader and return the new var name.
  // For codegen, just append '_transposed' for clarity.
  return `${originalVar}_transposed`;
}

/**
 * Generate JavaScript code for a WebNN conv2d operation from TFLite node info.
 * Keeps both NCHW and NHWC layout logic.
 */
export function Conv2D(
  node: any,
  toJsVarName: (name: string) => string,
  options: { [key: string]: any } = {}
): string {
  const nhwc = !!options.nhwc;
  const inputVars = getInputVars(node, toJsVarName);
  const outputVars = getOutputVars(node, toJsVarName);

  // Strides
  let strides = getAttrValue(node, 'strides', undefined);
  if (!strides) {
    const stride_h = getAttrValue(node, 'stride_h', 1);
    const stride_w = getAttrValue(node, 'stride_w', 1);
    strides = [stride_h, stride_w];
  }
  const strides_js = `[${strides.map((s: any) => String(Number(s))).join(', ')}]`;

  // Pads
  let pads = getAttrValue(node, 'pads', undefined);
  let pads_js = '[0, 0, 0, 0]';
  let paddingType = getAttrValue(node, 'padding', undefined);
  if (!pads && paddingType) {
    pads_js = paddingType === 'VALID' ? '[0, 0, 0, 0]' : `'${paddingType}'`;
  } else if (Array.isArray(pads) && pads.length === 4) {
    // ONNX: [top, left, bottom, right] -> WebNN: [top, bottom, left, right]
    const pads_webnn = [pads[0], pads[2], pads[1], pads[3]];
    pads_js = `[${pads_webnn.map((p: any) => String(Number(p))).join(', ')}]`;
  }

  // Dilations
  let dilations = getAttrValue(node, 'dilations', undefined);
  if (!dilations) {
    const dilation_h = getAttrValue(node, 'dilation_h_factor', 1);
    const dilation_w = getAttrValue(node, 'dilation_w_factor', 1);
    dilations = [dilation_h, dilation_w];
  }
  const dilations_js = `[${dilations.map((d: any) => String(Number(d))).join(', ')}]`;

  // Groups
  let groups = getAttrValue(node, 'group', 1);
  const groups_js = String(Number(groups));

  // Bias input (optional)
  const biasVar = inputVars.length > 2 ? inputVars[2] : undefined;

  // Extract filter shape
  const filterShape = getShape(node, 1, nhwc);

  // Determine inputLayout and filterLayout
  let inputLayout = nhwc ? "'nhwc'" : "'nchw'";
  let filterLayout = "'oihw'";
  let filterVarName = inputVars[1];

  // Detect depthwise conv
  let isDepthwise = false;
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
  const optionsArr: string[] = [
    `strides: ${strides_js}`,
    `padding: ${pads_js}`,
    `dilations: ${dilations_js}`,
    `groups: ${groups_js}`
  ];

  if (biasVar) optionsArr.push(`bias: ${biasVar}`);
  if (filterLayout) optionsArr.push(`filterLayout: ${filterLayout}`);
  if (inputLayout) optionsArr.push(`inputLayout: ${inputLayout}`);
  if (node.name) optionsArr.push(`label: '${node.name}'`);

  return `
    const ${outputVars[0]} = builder.conv2d(
      ${inputVars[0]}, ${filterVarName},
      {
        ${optionsArr.join(',\n        ')}
      }
    );`;
}