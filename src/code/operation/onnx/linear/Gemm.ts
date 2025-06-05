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
  toJsVarName: (name: string) => string,
  options: { nhwc?: boolean } = {}
): string {
  const inputVars = getInputVars(node, toJsVarName);
  const outputVars = getOutputVars(node, toJsVarName);
  const attrs: any[] = node.attributes || [];
  const nhwc = !!options.nhwc;

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

  let transA = Number(attrDict['transA']?.value?.value ?? 0);
  let transB = Number(attrDict['transB']?.value?.value ?? 0);

  let inputA = inputVars[0];
  // If NHWC and input is 4D, flatten in NHWC order
  const inputShape = node.inputs?.[0]?.value?.[0]?.type?.shape?.dimensions;
  if (nhwc && Array.isArray(inputShape) && inputShape.length === 4) {
    // Flatten NHWC: [N, H, W, C] -> [N, H*W*C]
    const flattenShape = [inputShape[0], inputShape[1] * inputShape[2] * inputShape[3]];
    const flatVar = `${inputA}_flat`;
    inputA = flatVar;
    // Insert flatten code before gemm
    return `
      const ${flatVar} = builder.reshape(${inputVars[0]}, [${flattenShape.join(', ')}]);
      const ${outputVars[0]} = builder.gemm(
        ${flatVar},
        ${inputVars[1]},
        {
          alpha: ${alpha},
          beta: ${beta},
          aTranspose: ${Boolean(transA)},
          bTranspose: true${inputVars.length > 2 ? `, C: ${inputVars[2]}` : ''}
        }
      );`;
  }

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