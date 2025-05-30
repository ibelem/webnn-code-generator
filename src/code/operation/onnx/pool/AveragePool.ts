/**
 * Generate JavaScript code for a WebNN averagePool2d operation from ONNX AveragePool node info.
 * https://www.w3.org/TR/webnn/#api-mlgraphbuilder-pool2d-average
 */

import { getNonEmptyStringAroundNewline } from '../../../../utils';

export function AveragePool(
  node: any,
  toJsVarName: (name: string) => string
): string {
  const inputs: string[] = node.inputs?.map((i: any) => getNonEmptyStringAroundNewline(i.value?.[0].name)) || [];
  const outputs: string[] = node.outputs?.map((o: any) => getNonEmptyStringAroundNewline(o.value?.[0].name)) || [];
  const inputVar = toJsVarName(inputs[0]);
  const outputVar = toJsVarName(outputs[0]);

  // Extract attributes
  const attrDict = Object.fromEntries(
    (node.attributes || []).map((a: any) => [a.name, a])
  );

  const kernelShape = attrDict['kernel_shape']?.value;
  const strides = attrDict['strides']?.value;
  const pads = attrDict['pads']?.value;

  const opts: string[] = [];
  if (kernelShape) opts.push(`windowDimensions: [${kernelShape.join(', ')}]`);
  if (strides) opts.push(`strides: [${strides.join(', ')}]`);
  if (pads) opts.push(`padding: [${pads.join(', ')}]`);
  const options = opts.length ? `, { ${opts.join(', ')} }` : '';

  return `
    const ${outputVar} = builder.averagePool2d(
      ${inputVar}${options}
    );`;
}