import {
  getInputVars,
  getOutputVars
} from '../../operation-utils';

/**
 * Generate JavaScript code for a WebNN resample2d operation from ONNX Resize node info.
 * https://www.w3.org/TR/webnn/#api-mlgraphbuilder-resample2d-method
 * https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/webnn/builders/impl/resize_op_builder.cc
 */
export function Resize(
  node: any,
  toJsVarName: (name: string) => string,
  options: { [key: string]: any } = {}
): string {
  const nhwc = !!options.nhwc;
  const inputVars = getInputVars(node, toJsVarName);
  const outputVars = getOutputVars(node, toJsVarName);
  const attrs: any[] = node.attributes || [];

  // Default mode is 'nearest'
  let mode = 'nearest';
  for (const attr of attrs) {
    if (attr.name === 'mode' && typeof attr.value === 'string') {
      mode = attr.value.toLowerCase();
    }
  }

  // Map ONNX mode to WebNN mode
  let webnn_mode = 'nearest-neighbor';
  if (mode === 'linear') {
    webnn_mode = 'linear';
  } else if (mode === 'cubic') {
    throw new Error('WebNN does not support cubic mode for Resize.');
  }

  // Helper to extract initializer array from input index
  function getInitializerArr(idx: number): number[] | undefined {
    if (node.inputs.length > idx && node.inputs[idx]?.value?.[0]?.initializer) {
      const init = node.inputs[idx].value[0].initializer;
      const arr = Object.keys(init.values)
        .sort((a, b) => Number(a) - Number(b))
        .map(k => init.values[k]);
      return arr;
    }
    return undefined;
  }

  // sizes: usually input[3]
  let sizes_js: string | undefined = undefined;
  const sizesArr = getInitializerArr(3);
  if (sizesArr && sizesArr.length >= 4) {
    // Only support resizing spatial dims (H, W)
    sizes_js = `[${parseInt(sizesArr[2].toString())}, ${parseInt(sizesArr[3].toString())}]`;
  } else if (node.inputs.length > 3 && node.inputs[3]) {
    sizes_js = inputVars[3];
  }

  // scales: usually input[2]
  let scales_js: string | undefined = undefined;
  const scalesArr = getInitializerArr(2);
  if (scalesArr && scalesArr.length >= 4) {
    // Only support scaling spatial dims (H, W)
    scales_js = `[${parseFloat(scalesArr[2].toString())}, ${parseFloat(scalesArr[3].toString())}]`;
  } else if (node.inputs.length > 2 && node.inputs[2]) {
    scales_js = inputVars[2];
  }

  // Axes: default per spec, or use [1,2] for NHWC, [2,3] for NCHW
  let axes_js = nhwc ? '[1, 2]' : '[2, 3]';

  // Add label for debugging
  const labelOpt = node.name ? `label: '${node.name}', ` : '';

  // Build options
  const opts: string[] = [`mode: '${webnn_mode}'`];
  if (sizes_js) {
    opts.push(`sizes: ${sizes_js}`);
  }
  if (scales_js) {
    opts.push(`scales: ${scales_js}`);
  }
  opts.push(`axes: ${axes_js}`);
  if (labelOpt) {
    opts.push(labelOpt.slice(0, -2)); // Remove trailing comma and space
  }

  return `
    const ${outputVars[0]} = builder.resample2d(
      ${inputVars[0]},
      {
        ${opts.join(',\n    ')}
      }
    );`;
}