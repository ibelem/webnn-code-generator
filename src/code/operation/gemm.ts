/**
 * Generate JavaScript code for a WebNN gemm operation from ONNX Gemm node info.
 * @param node - The ONNX node object (with inputs, outputs, attributes)
 * @param toJsVarName - Function to convert ONNX names to JS variable names
 * @returns JavaScript code string for the gemm operation
 */
export function gemm_js(
  node: any,
  toJsVarName: (name: string) => string
): string {
  const inputs: string[] = node.inputs?.map((i: any) => i.value?.[0]?.name) || [];
  const outputs: string[] = node.outputs?.map((o: any) => o.value?.[0]?.name) || [];
  const attrs: any[] = node.attributes || [];

  // Map attribute array to a dictionary by name
  const attrDict: Record<string, any> = {};
  for (const attr of attrs) {
    attrDict[attr.name] = attr;
  }

  const inputVars = inputs.map(toJsVarName);
  const outputVar = toJsVarName(outputs[0]);

  function formatFloat(val: any, type: string | undefined): string {
    if (type === 'float16' || type === 'float32' || type === 'float64') {
      return `${Number(val)}.0`;
    }
    return `${Number(val)}`;
  }

  const alphaType = attrDict['alpha']?.type;
  const betaType = attrDict['beta']?.type;
  const alphaVal = attrDict['alpha']?.value?.value ?? 1.0;
  const betaVal = attrDict['beta']?.value?.value ?? 1.0;

  const alpha = formatFloat(alphaVal, alphaType);
  const beta = formatFloat(betaVal, betaType);

  const transA = Number(attrDict['transA']?.value?.value ?? 0);
  const transB = Number(attrDict['transB']?.value?.value ?? 0);

  // WebNN: builder.gemm(A, B, options)
  // If C is present, pass as 'C' in options
  const options: string[] = [
    `alpha: ${alpha}`,
    `beta: ${beta}`,
    `aTranspose: ${Boolean(transA)}`,
    `bTranspose: ${Boolean(transB)}`
  ];
  if (inputVars.length > 2) {
    options.push(`C: ${inputVars[2]}`);
  }

  return `
    const ${outputVar} = builder.gemm(
      ${inputVars[0]},
      ${inputVars[1]},
      {
        ${options.join(', ')}
      }
    );`;
}