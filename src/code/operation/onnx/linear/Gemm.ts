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

  // Get transA/transB from ONNX attributes
  let transA = Number(attrDict['transA']?.value?.value ?? 0);
  let transB = Number(attrDict['transB']?.value?.value ?? 0);

  // Get input and weight shapes
  const inputShape = node.inputs?.[0]?.value?.[0]?.type?.shape?.dimensions;
  const weightShape = node.inputs?.[1]?.value?.[0]?.type?.shape?.dimensions;

  // Debug string to include in output
  let debugComment = '';
  
  let inputA = inputVars[0];
  
  // Special handling for NHWC 4D input (typically from Conv)
  if (nhwc && Array.isArray(inputShape) && inputShape.length === 4) {
    // For NHWC: [N,H,W,C] -> [N, H*W*C]
    const flattenShape = [inputShape[0], inputShape[1] * inputShape[2] * inputShape[3]];
    const flatVar = `${inputA}_flat`;
    inputA = flatVar;
    
    // For MobileNetV2 and similar networks, when converting from NCHW to NHWC:
    // - Original weights are [out_channels, in_channels]
    // - NHWC needs weights as [out_channels, in_channels] 
    // - We need to conditionally set bTranspose depending on the model structure
    
    // In MobileNetV2's case, for conv followed by gemm, we typically need:
    // 1. Keep original transB value for "normal" shape weights
    // 2. Toggle transB value for NHWC adjustment
    
    // Only modify transB if we're in NHWC mode
    if (nhwc) {
      // For MobileNetV2 on CPU with NHWC, we need transB = !transB
      // This flips the transpose flag to fix the dimension mismatch
      transB = transB === 0 ? 1 : 0;
    }
    
    debugComment = `\n      // Input shape: [${inputShape}] -> [${flattenShape}]\n      // Weight shape: [${weightShape}]\n      // NHWC mode: ${nhwc}, Original transB: ${attrDict['transB']?.value?.value ?? 0}, Using transB: ${transB}\n`;
    
    // Insert flatten code before gemm
    return `${debugComment}
      const ${flatVar} = builder.reshape(${inputVars[0]}, [${flattenShape.join(', ')}]);
      const ${outputVars[0]} = builder.gemm(
        ${flatVar},
        ${inputVars[1]},
        {
          alpha: ${alpha},
          beta: ${beta},
          aTranspose: ${Boolean(transA)},
          bTranspose: ${Boolean(transB)}${inputVars.length > 2 ? `,\n          C: ${inputVars[2]}` : ''}
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
        ${opts.join(',\n        ')}
      }
    );`;
}