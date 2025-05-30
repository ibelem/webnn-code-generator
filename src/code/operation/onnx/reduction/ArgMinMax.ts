/**
 * Generate JavaScript code for a WebNN argMax or argMin operation from ONNX node info.
 * https://www.w3.org/TR/webnn/#api-mlgraphbuilder-argminmax
 */

import { getNonEmptyStringAroundNewline, mlOperandDataType } from '../../../../utils';

export function ArgMaxMin(
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
  let keepDims = true; // ONNX default is keepdims=1 (true)
  for (const attr of attrs) {
    if (attr.name === 'axis') {
      // Handle ONNX JSON format: value: { type: "bigint", value: "-1" }
      if (attr.value && typeof attr.value.value === 'string') {
        axis = Number(attr.value.value);
      } else if (typeof attr.value === 'number') {
        axis = attr.value;
      }
    }
    if (attr.name === 'keepdims') {
      // ONNX default is 1 (true), 0 (false)
      if (attr.value && typeof attr.value.value === 'string') {
        keepDims = Number(attr.value.value) !== 0;
      } else if (typeof attr.value === 'number') {
        keepDims = attr.value !== 0;
      }
    }
  }

  // Handle negative axis
  if (axis < 0) {
    const inputShape = node.inputs?.[0]?.value?.[0]?.type?.shape?.dimensions || [];
    axis = inputShape.length + axis;
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

export const ArgMax = (node: any, toJsVarName: (name: string) => string) =>
  ArgMaxMin(node, toJsVarName, 'argMax');

export const ArgMin = (node: any, toJsVarName: (name: string) => string) =>
  ArgMaxMin(node, toJsVarName, 'argMin');