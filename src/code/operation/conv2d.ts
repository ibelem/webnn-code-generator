/**
 * Generate JavaScript code for a WebNN conv2d operation from ONNX node info.
 * @param node - The ONNX node object (with inputs, outputs, attributes)
 * @param toJsVarName - Function to convert ONNX names to JS variable names
 * @returns JavaScript code string for the conv2d operation
 */
import {  getNonEmptyStringAroundNewline  } from '../../utils';

/**
 * TFLite model for getting Padding values from SAME and VALID
 * @param input 
 * @param kernel 
 * @param stride 
 * @param dilation 
 * @returns 
 */
// function calcSamePadding(input: number, kernel: number, stride: number, dilation: number = 1) {
//   const effectiveKernel = (kernel - 1) * dilation + 1;
//   const out = Math.ceil(input / stride);
//   const pad = Math.max(0, (out - 1) * stride + effectiveKernel - input);
//   const padBefore = Math.floor(pad / 2);
//   const padAfter = pad - padBefore;
//   return [padBefore, padAfter];
// }

// function calcSamePadding2D(input: number[], kernel: number[], stride: number[], dilation: number[] = [1, 1]) {
//   const effectiveKernel = [(kernel[0] - 1) * dilation[0] + 1, (kernel[1] - 1) * dilation[1] + 1];
//   const out = [Math.ceil(input[0] / stride[0]), Math.ceil(input[1] / stride[1])];
//   const pad = [
//     Math.max(0, (out[0] - 1) * stride[0] + effectiveKernel[0] - input[0]),
//     Math.max(0, (out[1] - 1) * stride[1] + effectiveKernel[1] - input[1])
//   ];
//   const padBefore = [Math.floor(pad[0] / 2), Math.floor(pad[1] / 2)];
//   const padAfter = [pad[0] - padBefore[0], pad[1] - padBefore[1]];
//   return [padBefore, padAfter];
// }

/**
 * For VALID padding, the padding is always zero
 * TFLite model for getting Padding values from SAME and VALID
 * https://www.w3.org/TR/webnn/#dictdef-mlconv2doptions
 */
function calcValidPadding2D() {
  return [0, 0, 0, 0]; // [beginningHeight, endingHeight, beginningWidth, endingWidth]
}

export function conv2d_js(
  node: any,
  toJsVarName: (name: string) => string
): string {
  const inputs: string[] = node.inputs?.map((i: any) => getNonEmptyStringAroundNewline(i.value?.[0].name)) || [];
  const outputs: string[] = node.outputs?.map((o: any) => getNonEmptyStringAroundNewline(o.value?.[0].name)) || [];
  const attrs: any[] = node.attributes || [];

  // Map attribute array to a dictionary by name
  const attrDict: Record<string, any> = {};
  for (const attr of attrs) {
    attrDict[attr.name] = attr;
  }

  const inputVars = inputs.map(toJsVarName);
  const outputVar = toJsVarName(outputs[0]);

  // Handle strides (ONNX: strides, TFLite: stride_w/stride_h)
  let strides_js = 'undefined';
  let strides = attrDict['strides']?.value?.value;
  if (!strides && (attrDict['stride_w'] || attrDict['stride_h'])) {
    // TFLite style
    const stride_w = Number(attrDict['stride_w']?.value ?? 1);
    const stride_h = Number(attrDict['stride_h']?.value ?? 1);
    strides = [stride_h, stride_w];
  }
  if (Array.isArray(strides) && strides.length === 2) {
    strides_js = `[${strides.map((s: any) => String(Number(s))).join(', ')}]`;
  }

  // Handle padding (ONNX: pads, TFLite: padding)
  let pads_js = 'undefined';
  let pads = attrDict['pads']?.value?.value;
  if (!pads && attrDict['padding']?.value) {
    // TFLite style: 'SAME' or 'VALID'
    const padType = attrDict['padding'].value;
    if (padType === 'VALID') {
      const tfliteValidPadding = calcValidPadding2D().map((p: any) => String(Number(p))).join(', ');
      pads_js = `[${tfliteValidPadding}]`;
    } else if (padType === 'SAME') {
      pads_js = `'${padType}'`;
    }
  } else if (Array.isArray(pads) && pads.length === 4) {
    // ONNX: [top, left, bottom, right] -> WebNN: [top, bottom, left, right]
    const pads_webnn = [pads[0], pads[2], pads[1], pads[3]];
    pads_js = `[${pads_webnn.map((p: any) => String(Number(p))).join(', ')}]`;
  }

  // Handle dilations (ONNX: dilations, TFLite: dilation_w_factor/dilation_h_factor)
  let dilations_js = 'undefined';
  let dilations = attrDict['dilations']?.value?.value;
  if (!dilations && (attrDict['dilation_w_factor'] || attrDict['dilation_h_factor'])) {
    // TFLite style
    const dilation_w = Number(attrDict['dilation_w_factor']?.value ?? 1);
    const dilation_h = Number(attrDict['dilation_h_factor']?.value ?? 1);
    dilations = [dilation_h, dilation_w];
  }
  if (Array.isArray(dilations) && dilations.length === 2) {
    dilations_js = `[${dilations.map((d: any) => String(Number(d))).join(', ')}]`;
  }

  // Handle groups (ONNX: group, TFLite: not used, default to 1)
  const groups = attrDict['group']?.value?.value ?? 1;
  const groups_js = String(Number(groups));

  // Bias input (optional)
  const biasVar = inputVars.length > 2 ? inputVars[2] : 'undefined';

  return `
    const ${outputVar} = builder.conv2d(
      ${inputVars[0]}, ${inputVars[1]},
      {
        bias: ${biasVar},
        strides: ${strides_js},
        padding: ${pads_js},
        dilations: ${dilations_js},
        groups: ${groups_js}
      }
    );`;
}