import {
  getInputVars,
  getOutputVars
} from '../../operation-utils';

/**
 * Generate JavaScript code for a WebNN l2Pool2d operation from ONNX LpPool node info.
 * https://www.w3.org/TR/webnn/#api-mlgraphbuilder-pool2d-l2
 * https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/webnn/builders/impl/pool_op_builder.cc
 * Only supports p=2 (L2 norm), as required by WebNN.
 */
export function LpPool(
  node: any,
  toJsVarName: (name: string) => string,
  options: { nhwc?: boolean } = {}
): string {
  const inputVars = getInputVars(node, toJsVarName);
  const outputVars = getOutputVars(node, toJsVarName);
  const attrs: any[] = node.attributes || [];
  const attrDict: Record<string, any> = {};
  for (const attr of attrs) attrDict[attr.name] = attr;

  // Only support p=2 (L2 norm)
  const p = attrDict['p']?.value ?? 2;
  if (p !== 2) {
    return `// Only L2 pooling (p=2) is supported by WebNN.`;
  }

  const kernelShape = attrDict['kernel_shape']?.value;
  const strides = attrDict['strides']?.value;
  const pads = attrDict['pads']?.value;
  const ceilMode = attrDict['ceil_mode']?.value ?? 0;
  const nhwc = !!options.nhwc;

  // WebNN expects [beginH, endH, beginW, endW], ONNX is [beginH, beginW, endH, endW]
  let paddingStr = '';
  if (pads && pads.length === 4) {
    paddingStr = `padding: [${pads[0]}, ${pads[2]}, ${pads[1]}, ${pads[3]}]`;
  }

  const opts: string[] = [];
  if (kernelShape) opts.push(`windowDimensions: [${kernelShape.join(', ')}]`);
  if (paddingStr) opts.push(paddingStr);
  if (strides) opts.push(`strides: [${strides.join(', ')}]`);
  opts.push(`layout: '${nhwc ? 'nhwc' : 'nchw'}'`);
  opts.push(`roundingType: '${ceilMode ? 'ceil' : 'floor'}'`);
  if (node.name) opts.push(`label: '${node.name}'`);

  return `
    const ${outputVars[0]} = builder.l2Pool2d(
      ${inputVars[0]},
      {
        ${opts.join(',\n    ')}
      }
    );`;
}