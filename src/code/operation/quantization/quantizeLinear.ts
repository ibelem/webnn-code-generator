import { getNonEmptyStringAroundNewline } from '../../../utils';

/**
 * Generate JavaScript code for WebNN quantizeLinear operation from ONNX node info.
 * https://www.w3.org/TR/webnn/#api-mlgraphbuilder-quantizelinear
 */
export function quantizeLinear(
  node: any,
  toJsVarName: (name: string) => string
): string {
  const inputs: any[] = node.inputs || [];
  const outputs: any[] = node.outputs || [];
  const attrs: any[] = node.attributes || [];
  const inputVars = inputs.map(i => getNonEmptyStringAroundNewline(i.value?.[0]?.name)).map(toJsVarName);
  const outputVar = toJsVarName(getNonEmptyStringAroundNewline(outputs[0]?.value?.[0]?.name));

  // Get input/scale/zeroPoint shapes and dtypes
  const inputShape = inputs[0]?.value?.[0]?.type?.shape?.dimensions || [];
  const scaleShape = inputs[1]?.value?.[0]?.type?.shape?.dimensions || [];
  const outputDtype = outputs[0]?.value?.[0]?.type?.dataType || 'uint8';

  // Default axis is 1 for ONNX QuantizeLinear
  let axis = 1;
  for (const attr of attrs) {
    if (attr.name === 'axis') {
      if (attr.value && typeof attr.value.value === 'string') {
        axis = Number(attr.value.value);
      } else if (typeof attr.value === 'number') {
        axis = attr.value;
      }
      break;
    }
  }

  // Inline reshape for scale if needed
  let scaleExpr = inputVars[1];
  if (scaleShape.length !== inputShape.length) {
    const newShape = Array(inputShape.length).fill(1);
    if (scaleShape.length === 1) {
      newShape[axis] = scaleShape[0];
    }
    scaleExpr = `builder.reshape(${inputVars[1]}, [${newShape.join(', ')}])`;
  }

  // Inline reshape or constant for zeroPoint
  let zeroPointExpr: string;
  if (inputs.length > 2 && inputs[2]?.value?.[0]) {
    const zpShape = inputs[2]?.value?.[0]?.type?.shape?.dimensions || [];
    if (zpShape.length !== inputShape.length) {
      const newShape = Array(inputShape.length).fill(1);
      if (zpShape.length === 1) {
        newShape[axis] = zpShape[0];
      }
      zeroPointExpr = `builder.reshape(${inputVars[2]}, [${newShape.join(', ')}])`;
    } else {
      zeroPointExpr = inputVars[2];
    }
  } else {
    // If zeroPoint is not present, create a constant zero of same shape as scale (after expansion)
    const zpShape = scaleShape.length === inputShape.length
      ? scaleShape
      : (() => {
          const arr = Array(inputShape.length).fill(1);
          if (scaleShape.length === 1) arr[axis] = scaleShape[0];
          return arr;
        })();
    const zpElemCount = zpShape.reduce((a: number, b: number) => a * (typeof b === 'number' ? b : 1), 1) || 1;
    const typedArrayCtor = outputDtype === 'uint8' ? 'Uint8Array'
      : outputDtype === 'int8' ? 'Int8Array'
      : outputDtype === 'uint32' ? 'Uint32Array'
      : 'Int32Array';
    zeroPointExpr = `builder.constant(
      {dataType: '${outputDtype}', shape: [${zpShape.join(', ')}]},
      new ${typedArrayCtor}([${Array(zpElemCount).fill(0).join(', ')}])
    )`;
  }

  // Validation
  const allowedInputTypes = ['float32', 'float16'];
  const allowedZeroPointTypes = ['uint8', 'int8', 'uint32', 'int32'];
  if (!allowedInputTypes.includes(inputs[0]?.value?.[0]?.type?.dataType)) {
    throw new Error('quantizeLinear input must be float32 or float16');
  }
  if (!allowedInputTypes.includes(inputs[1]?.value?.[0]?.type?.dataType)) {
    throw new Error('quantizeLinear scale must be float32 or float16');
  }
  if (!allowedZeroPointTypes.includes(outputDtype)) {
    throw new Error('quantizeLinear zeroPoint/output must be uint8, int8, uint32, or int32');
  }

  return `
    const ${outputVar} = builder.quantizeLinear(
      ${inputVars[0]},
      ${scaleExpr},
      ${zeroPointExpr}
    );`;
}