import {
  getInputVars,
  getOutputVars
} from '../../operation-utils';

/**
 * Generate JavaScript code for a WebNN averagePool2d operation from ONNX AveragePool node info.
 * https://www.w3.org/TR/webnn/#api-mlgraphbuilder-pool2d-average
 */
export function AveragePool(
  node: any,
  toJsVarName: (name: string) => string
): string {
  const inputVars = getInputVars(node, toJsVarName);
  const outputVars = getOutputVars(node, toJsVarName);

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
  const optsString = opts.length ? `, { ${opts.join(', ')} }` : '';

  return `
    const ${outputVars[0]} = builder.averagePool2d(
      ${inputVars[0]}${optsString}
    );`;
}