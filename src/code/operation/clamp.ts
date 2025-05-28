/**
 * Generate JavaScript code for a WebNN clamp operation from ONNX Clip node info.
 * @param node - The ONNX node object (with inputs, outputs)
 * @param toJsVarName - Function to convert ONNX names to JS variable names
 * @returns JavaScript code string for the clamp operation
 */

/**
 * WebNN Specification: https://www.w3.org/TR/webnn/
 * https://www.w3.org/TR/webnn/#api-mlgraphbuilder-clamp
 */

import { getNonEmptyStringAroundNewline } from '../../utils';
export function clamp(
  node: any,
  toJsVarName: (name: string) => string
): string {
  const inputs: any[] = node.inputs || [];
  const outputs: string[] = node.outputs?.map((o: any) => getNonEmptyStringAroundNewline(o.value?.[0].name)) || [];
  const outputVar = toJsVarName(outputs[0]);

  // Helper to extract scalar from initializer
  function getScalarFromInitializer(inputIdx: number): string | undefined {
    const val = inputs[inputIdx]?.value?.[0];
    const dims = val?.initializer?.type?.shape?.dimensions;
    if (
      val?.initializer &&
      val.initializer.values &&
      (Array.isArray(dims) && (dims.length === 0 || (dims.length === 1 && dims[0] === 1)))
    ) {
      const keys = Object.keys(val.initializer.values);
      if (keys.length === 1) {
        return val.initializer.values[keys[0]].toString();
      }
    }
    return undefined;
  }

  const inputVar = toJsVarName(inputs[0]?.value?.[0]?.name);
  const minValue = inputs.length > 1
    ? getScalarFromInitializer(1) ?? toJsVarName(inputs[1]?.value?.[0]?.name)
    : 'undefined';
  const maxValue = inputs.length > 2
    ? getScalarFromInitializer(2) ?? toJsVarName(inputs[2]?.value?.[0]?.name)
    : 'undefined';

  return `
    const ${outputVar} = builder.clamp(
      ${inputVar},
      {
        minValue: ${minValue},
        maxValue: ${maxValue}
      }
    );`;
}