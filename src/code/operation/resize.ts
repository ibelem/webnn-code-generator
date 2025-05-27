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
    if (attr.name === 'mode' && attr.s) {
      try {
        mode = atob(attr.s).toLowerCase();
      } catch {
        mode = attr.s.toLowerCase();
      }
    }
  }

  // Map ONNX mode to WebNN mode
  let webnn_mode = 'nearest-neighbor';
  if (mode === 'linear') {
    webnn_mode = 'linear';
  } else if (mode === 'cubic') {
    throw new Error('WebNN does not support cubic mode for Resize.');
  }

  // Handle scales and sizes (try to inline if possible, else use variable)
  let scales_js = undefined;
  let sizes_js = undefined;

  // sizes: usually input[3]
  if (inputs.length > 3 && inputs[3]) {
    // Try to inline if available as initializer (should be resolved in your loader)
    // Here, just use the variable name for codegen
    sizes_js = toJsVarName(inputs[3]);
  }
  // scales: usually input[2]
  if (inputs.length > 2 && inputs[2]) {
    scales_js = toJsVarName(inputs[2]);
  }

  const options = [
    `mode: '${webnn_mode}'`,
    `scales: ${scales_js}`
  ];

  if(sizes_js) options.push(`sizes: ${sizes_js}`);

  if (nhwc) {
    options.push('axes: [1, 2]');
  }

  return `
    const ${outputVar} = builder.resample2d(
      ${inputVar},
      {
        ${options.join(',\n        ')}
      }
    );`;
}