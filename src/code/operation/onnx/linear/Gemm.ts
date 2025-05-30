import {
  getInputVars,
  getOutputVars
} from '../../operation-utils';

/**
 * Generate JavaScript code for a WebNN gemm operation from ONNX Gemm node info.
 * https://www.w3.org/TR/webnn/#api-mlgraphbuilder-gemm
 */
export function Gemm(
  node: any,
  toJsVarName: (name: string) => string
): string {
  const inputVars = getInputVars(node, toJsVarName);
  const outputVars = getOutputVars(node, toJsVarName);
  const attrs: any[] = node.attributes || [];

  // Map attribute array to a dictionary by name
  const attrDict: Record<string, any> = {};
  for (const attr of attrs) {
    attrDict[attr.name] = attr;
  }

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
  const opts: string[] = [
    `alpha: ${alpha}`,
    `beta: ${beta}`,
    `aTranspose: ${Boolean(transA)}`,
    `bTranspose: ${Boolean(transB)}`
  ];
  if (inputVars.length > 2) {
    opts.push(`C: ${inputVars[2]}`);
  }

  return `
    const ${outputVars[0]} = builder.gemm(
      ${inputVars[0]},
      ${inputVars[1]},
      {
        ${opts.join(', ')}
      }
    );`;
}