import {
  getInputVars,
  getOutputVars
} from '../../operation-utils';

/**
 * Generate JavaScript code for a WebNN clamp operation from ONNX Clip node info.
 * https://www.w3.org/TR/webnn/#api-mlgraphbuilder-clamp
 */
export function Clip(
  node: any,
  toJsVarName: (name: string) => string
): string {
  const inputVars = getInputVars(node, toJsVarName);
  const outputVars = getOutputVars(node, toJsVarName);

  // Helper to extract scalar from initializer
  function getScalarFromInitializer(inputIdx: number): string | undefined {
    const val = node.inputs?.[inputIdx]?.value?.[0];
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

  const minValue = node.inputs.length > 1
    ? getScalarFromInitializer(1) ?? inputVars[1]
    : 'undefined';
  const maxValue = node.inputs.length > 2
    ? getScalarFromInitializer(2) ?? inputVars[2]
    : 'undefined';

  return `
    const ${outputVars[0]} = builder.clamp(
      ${inputVars[0]},
      {
        minValue: ${minValue},
        maxValue: ${maxValue}
      }
    );`;
}