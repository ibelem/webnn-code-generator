/**
 * Generate JavaScript code for a WebNN averagePool2d operation from ONNX GlobalAveragePool node info.
 * @param node - The ONNX node object (with inputs, outputs)
 * @param toJsVarName - Function to convert ONNX names to JS variable names
 * @returns JavaScript code string for the averagePool2d operation
 */
export function averagePool2d_js(
  node: any,
  toJsVarName: (name: string) => string
): string {
  const inputs: string[] = node.inputs?.map((i: any) => i.value?.[0]?.name) || [];
  const outputs: string[] = node.outputs?.map((o: any) => o.value?.[0]?.name) || [];

  const inputVars = inputs.map(toJsVarName);
  const outputVar = toJsVarName(outputs[0]);

  return `
    const ${outputVar} = builder.averagePool2d(
      ${inputVars[0]}
    );`;
}