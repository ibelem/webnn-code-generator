import { getInputVars, getOutputVars, getAttrValue } from '../../operation-utils';

/**
 * Generate JavaScript code for a WebNN averagePool2d operation from ONNX AveragePool node info.
 * https://www.w3.org/TR/webnn/#api-mlgraphbuilder-pool2d-average
 * https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/webnn/builders/impl/pool_op_builder.cc
 */

export function AveragePool(
  node: any,
  toJsVarName: (name: string) => string,
  options: { nhwc?: boolean } = {}
): string {
  const inputVars = getInputVars(node, toJsVarName);
  const outputVars = getOutputVars(node, toJsVarName);
  const nhwc = !!options.nhwc;

  // Use the robust helper
  const kernelShape = getAttrValue(node, 'kernel_shape', [0, 0]);
  const pads = getAttrValue(node, 'pads', [0, 0, 0, 0]);
  const strides = getAttrValue(node, 'strides', [1, 1]);
  const dilations = getAttrValue(node, 'dilations', [1, 1]);
  const ceilMode = getAttrValue(node, 'ceil_mode', 0);
  const countIncludePad = getAttrValue(node, 'count_include_pad', 0);

  // WebNN expects [beginH, endH, beginW, endW], ONNX is [beginH, beginW, endH, endW]
  let webnnPads = [pads[0], pads[2], pads[1], pads[3]];

  let inputExpr = inputVars[0];
  let paddingOpt = `padding: [${webnnPads.join(', ')}]`;

  // Emulate count_include_pad=1 by explicit pad op
  if (countIncludePad === 1) {
    let beginPad, endPad;
    if (nhwc) {
      beginPad = [0, pads[0], pads[1], 0];
      endPad = [0, pads[2], pads[3], 0];
    } else {
      beginPad = [0, 0, pads[0], pads[1]];
      endPad = [0, 0, pads[2], pads[3]];
    }
    inputExpr = `builder.pad(${inputVars[0]}, [${beginPad.join(', ')}], [${endPad.join(', ')}])`;
    paddingOpt = ''; // Don't set padding in averagePool2d
  }

  const poolOpts: string[] = [
    `windowDimensions: [${kernelShape.join(', ')}]`,
    `strides: [${strides.join(', ')}]`,
    `dilations: [${dilations.join(', ')}]`,
    `roundingType: '${ceilMode === 1 ? 'ceil' : 'floor'}'`,
    `layout: '${nhwc ? 'nhwc' : 'nchw'}'`
  ];
  if (paddingOpt) poolOpts.push(paddingOpt);
  if (node.name) poolOpts.push(`label: '${node.name}'`);

  return `
    const ${outputVars[0]} = builder.averagePool2d(
      ${inputExpr},
      {
        ${poolOpts.join(',\n        ')} 
      }
    );`;
}