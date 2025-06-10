import {
  getInputVars,
  getOutputVars,
  getShape
} from '../../operation-utils';
import { mlOperandDataType } from '../../../../utils';

/**
 * Generate JavaScript code for a WebNN argMax or argMin operation from ONNX node info.
 * https://www.w3.org/TR/webnn/#api-mlgraphbuilder-argminmax
 * https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/webnn/builders/impl/argmax_min_op_builder.cc
 */
function ArgMinMax(
  node: any,
  toJsVarName: (name: string) => string,
  options: { [key: string]: any } = {}
): string {
  const nhwc = !!options.nhwc;
  const inputVars = getInputVars(node, toJsVarName);
  const outputVars = getOutputVars(node, toJsVarName);
  const inputShape = getShape(node, 0, nhwc);

  // Default axis is 0 for ONNX ArgMax/ArgMin
  let axis = 0;
  let keepDims = true; // ONNX default is keepdims=1 (true)
  for (const attr of node.attributes || []) {
    if (attr.name === 'axis') {
      if (attr.value && typeof attr.value.value === 'string') {
        axis = Number(attr.value.value);
      } else if (typeof attr.value === 'number') {
        axis = attr.value;
      }
    }
    if (attr.name === 'keepdims') {
      if (attr.value && typeof attr.value.value === 'string') {
        keepDims = Number(attr.value.value) !== 0;
      } else if (typeof attr.value === 'number') {
        keepDims = attr.value !== 0;
      }
    }
  }

  // Handle negative axis
  if (axis < 0) {
    axis = inputShape.length + axis;
  }

  // Set outputDataType to 'int64' by default, fallback to 'int32' if not supported
  let outputDataType = 'int64';
  if (node.outputs?.[0]?.value?.[0]?.type?.dataType) {
    const onnxType = node.outputs[0].value[0].type.dataType;
    outputDataType = mlOperandDataType(onnxType);
  }
  // Optionally fallback to int32 if int64 is not supported by backend (user can override if needed)

  const opType = options.opType;
  const labelOpt = node.name ? `, label: '${node.name}'` : '';

  return `
    const ${outputVars[0]} = builder.${opType}(
      ${inputVars[0]},
      ${axis},
      { keepDimensions: ${keepDims}, outputDataType: '${outputDataType}'${labelOpt} }
    );`;
}

export function ArgMax(
  node: any,
  toJsVarName: (name: string) => string,
  options: { [key: string]: any } = {}
): string {
  return ArgMinMax(node, toJsVarName, { ...options, opType: 'argMax' });
}

export function ArgMin(
  node: any,
  toJsVarName: (name: string) => string,
  options: { [key: string]: any } = {}
): string {
  return ArgMinMax(node, toJsVarName, { ...options, opType: 'argMin' });
}