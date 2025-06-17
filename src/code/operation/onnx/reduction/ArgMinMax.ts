import {
  getInputVars,
  getOutputVars,
  getShape,
  getAttrValue
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
  const inputVars = getInputVars(node, toJsVarName);
  const outputVars = getOutputVars(node, toJsVarName);

  // Default axis is 0 for ONNX ArgMax/ArgMin

  let axis = getAttrValue(node, 'axis', 0);
  let keepDims = !!getAttrValue(node, 'keepdims', 1);
  const selectLastIndex = !!getAttrValue(node, 'select_last_index', 0);
  if (selectLastIndex) {
    // If select_last_index is true, we need to set keepDims to false
    // because WebNN does not support this option.
    // ONNX ArgMax/ArgMin with select_last_index always returns a scalar.
    // so we set keepDims to false.
    keepDims = false;
  }

  // Get input rank for negative axis handling and validation
  let inputRank = 0;
  if (node.inputs && node.inputs.length > 0) {
    const shape = getShape(node, 0, false);
    if (Array.isArray(shape)) inputRank = shape.length;
  }

  // Resolve negative axis and validate
  if (axis < 0 && inputRank > 0) {
    axis = inputRank + axis;
  }
  if (inputRank > 0 && (axis < 0 || axis >= inputRank)) {
    throw new Error(`ArgMinMax: axis ${axis} is out of range for input rank ${inputRank}`);
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