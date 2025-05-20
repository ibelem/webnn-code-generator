/**
 * Generate JavaScript code for a WebNN clamp operation from ONNX Clip node info.
 * @param node - The ONNX node object (with inputs, outputs)
 * @param toJsVarName - Function to convert ONNX names to JS variable names
 * @returns JavaScript code string for the clamp operation
 */
export function clamp_js(
  node: any,
  toJsVarName: (name: string) => string
): string {
  const inputs: string[] = node.inputs?.map((i: any) => i.value?.[0].name) || [];
  const outputs: string[] = node.outputs?.map((o: any) => o.value?.[0].name) || [];

  const inputVars = inputs.map(toJsVarName);
  const outputVar = toJsVarName(outputs[0]);

  // ONNX Clip: input, min, max
  // WebNN clamp: builder.clamp(x, options)
  // options: {minValue, maxValue}
  const minValue = inputVars.length > 1 ? inputVars[1] : 'undefined';
  const maxValue = inputVars.length > 2 ? inputVars[2] : 'undefined';

  return `
    const ${outputVar} = builder.clamp(
      ${inputVars[0]},
      {
        minValue: ${minValue},
        maxValue: ${maxValue}
      }
    );`;
}