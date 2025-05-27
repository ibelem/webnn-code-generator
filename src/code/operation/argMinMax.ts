/**
 * Generate JavaScript code for a WebNN argMax or argMin operation from ONNX node info.
 * https://www.w3.org/TR/webnn/#api-mlgraphbuilder-argminmax
 */

import { getNonEmptyStringAroundNewline, mlOperandDataType } from '../../utils';

export function argMaxMin(
  node: any,
  toJsVarName: (name: string) => string,
  opType: 'argMax' | 'argMin'
): string {
  const inputs: string[] = node.inputs?.map((i: any) => getNonEmptyStringAroundNewline(i.value?.[0]?.name)) || [];
  const outputs: string[] = node.outputs?.map((o: any) => getNonEmptyStringAroundNewline(o.value?.[0]?.name)) || [];
  const attrs: any[] = node.attributes || [];
  const inputVar = toJsVarName(inputs[0]);
  const outputVar = toJsVarName(outputs[0]);

  // Default axis is 0 for ONNX ArgMax/ArgMin
  let axis = 0;
  let keepDims = false;
  for (const attr of attrs) {
    if (attr.name === 'axis') axis = Number(attr.i ?? 0);
    if (attr.name === 'keepdims') keepDims = !!attr.i;
  }

  // Set outputDataType to 'int32' as per WebNN default
  // Get output data type from node.outputs if available
  let outputDataType = 'int32';
  if (outputs.length > 0 && node.outputs[0]?.value?.[0]?.type?.dataType) {
    const onnxType = node.outputs[0].value[0].type.dataType;
    outputDataType = mlOperandDataType(onnxType);
  }

  return `
    const ${outputVar} = builder.${opType}(
      ${inputVar},
      ${axis},
      { keepDimensions: ${keepDims}, outputDataType: '${outputDataType}' }
    );`;
}

export const argMax = (node: any, toJsVarName: (name: string) => string) =>
  argMaxMin(node, toJsVarName, 'argMax');

export const argMin = (node: any, toJsVarName: (name: string) => string) =>
  argMaxMin(node, toJsVarName, 'argMin');