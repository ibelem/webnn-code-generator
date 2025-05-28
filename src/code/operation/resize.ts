/**
 * Generate JavaScript code for a WebNN resample2d operation from ONNX Resize node info.
 * https://www.w3.org/TR/webnn/#api-mlgraphbuilder-resample2d-method
 */

import { getNonEmptyStringAroundNewline } from '../../utils';

export function resize(
  node: any,
  toJsVarName: (name: string) => string,
  nhwc: boolean = false
): string {
  const inputs: string[] = node.inputs?.map((i: any) => getNonEmptyStringAroundNewline(i.value?.[0]?.name)) || [];
  const outputs: string[] = node.outputs?.map((o: any) => getNonEmptyStringAroundNewline(o.value?.[0]?.name)) || [];
  const attrs: any[] = node.attributes || [];
  const inputVar = toJsVarName(inputs[0]);
  const outputVar = toJsVarName(outputs[0]);

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

  // Handle scales and sizes
  let scales_js: string | undefined = undefined;
  let sizes_js: string | undefined = undefined;

  // Helper to extract initializer array from input index
  function getInitializerArr(idx: number): number[] | undefined {
    if (inputs.length > idx && node.inputs[idx]?.value?.[0]?.initializer) {
      const init = node.inputs[idx].value[0].initializer;
      const arr = Object.keys(init.values)
        .sort((a, b) => Number(a) - Number(b))
        .map(k => init.values[k]);
      return arr;
    }
    return undefined;
  }

  // sizes: usually input[3]
  const sizesArr = getInitializerArr(3);
  if (sizesArr && sizesArr.length >= 4) {
    sizes_js = `[${parseInt(sizesArr[2].toString())}, ${parseInt(sizesArr[3].toString())}]`;
  } else if (inputs.length > 3 && inputs[3]) {
    sizes_js = toJsVarName(inputs[3]);
  }

  // scales: usually input[2]
  const scalesArr = getInitializerArr(2);
  if (scalesArr && scalesArr.length >= 4) {
    scales_js = `[${parseFloat(scalesArr[2].toString())}, ${parseFloat(scalesArr[3].toString())}]`;
  } else if (inputs.length > 2 && inputs[2]) {
    scales_js = toJsVarName(inputs[2]);
  }

  // Axes: default per spec
  let axes_js = nhwc ? '[1, 2]' : '[2, 3]';

  // Build options
  const options: string[] = [`mode: '${webnn_mode}'`];
  if (sizes_js) {
    options.push(`sizes: ${sizes_js}`);
  } else {
    options.push(`sizes: undefined`);
  } 
  
  if (scales_js) {
    options.push(`scales: ${scales_js}`);
  } else {
    options.push('scales: [1.0, 1.0]');
  }
  options.push(`axes: ${axes_js}`);

  return `
    const ${outputVar} = builder.resample2d(
      ${inputVar},
      {
        ${options.join(',\n        ')}
      }
    );`;
}