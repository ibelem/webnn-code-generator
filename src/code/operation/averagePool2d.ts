/**
 * Generate JavaScript code for a WebNN averagePool2d operation from ONNX GlobalAveragePool node info.
 * @param node - The ONNX node object (with inputs, outputs)
 * @param toJsVarName - Function to convert ONNX names to JS variable names
 * @returns JavaScript code string for the averagePool2d operation
 */

/**
 * WebNN Specification: https://www.w3.org/TR/webnn/
 * https://www.w3.org/TR/webnn/#api-mlgraphbuilder-pool2d-average
 */

import { getNonEmptyStringAroundNewline } from '../../utils';

export function averagePool2d_js(
  node: any,
  toJsVarName: (name: string) => string
): string {
  const inputs: string[] = node.inputs?.map((i: any) => getNonEmptyStringAroundNewline(i.value?.[0].name)) || [];
  const outputs: string[] = node.outputs?.map((o: any) => getNonEmptyStringAroundNewline(o.value?.[0].name)) || [];
  const inputVar = toJsVarName(inputs[0]);
  const outputVar = toJsVarName(outputs[0]);
  const type = node.type?.name || '';

  // Extract attributes
  const attrDict = Object.fromEntries(
    (node.attributes || []).map((a: any) => [a.name, a])
  );

  let options = '';

  if (type === 'GlobalAveragePool') {
    // For GlobalAveragePool, kernel size should be set to input spatial dims
    // (Assume input shape is [N, C, H, W] or [N, H, W, C], depending on layout)
    // Here, we just omit kernelShape and let backend handle it as global
    options = '';
  } else {
    // For AveragePool, handle kernelShape, strides, pads, etc.
    const kernelShape = attrDict['kernel_shape']?.value;
    const strides = attrDict['strides']?.value;
    const pads = attrDict['pads']?.value;

    const opts: string[] = [];
    if (kernelShape) opts.push(`windowDimensions: [${kernelShape.join(', ')}]`);
    if (strides) opts.push(`strides: [${strides.join(', ')}]`);
    if (pads) opts.push(`padding: [${pads.join(', ')}]`);
    options = opts.length ? `, { ${opts.join(', ')} }` : '';
  }

  return `
    const ${outputVar} = builder.averagePool2d(
      ${inputVar}${options}
    );`;
}