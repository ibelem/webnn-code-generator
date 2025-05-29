import { getNonEmptyStringAroundNewline } from '../../utils';

/**
 * Generate JavaScript code for WebNN matmul operation from ONNX node info.
 * https://www.w3.org/TR/webnn/#api-mlgraphbuilder-matmul
 */
export function matmul(
  node: any,
  toJsVarName: (name: string) => string
): string {
  const inputs: any[] = node.inputs || [];
  const outputs: any[] = node.outputs || [];
  const inputVars = inputs.map(i => getNonEmptyStringAroundNewline(i.value?.[0]?.name)).map(toJsVarName);
  const outputVar = toJsVarName(getNonEmptyStringAroundNewline(outputs[0]?.value?.[0]?.name));

  // Validation (optional, can be removed for pure codegen)
  const allowedTypes = ['float32', 'float16'];
  const aType = inputs[0]?.value?.[0]?.type?.dataType;
  const bType = inputs[1]?.value?.[0]?.type?.dataType;
  const aShape = inputs[0]?.value?.[0]?.type?.shape?.dimensions || [];
  const bShape = inputs[1]?.value?.[0]?.type?.shape?.dimensions || [];
  if (!allowedTypes.includes(aType) || !allowedTypes.includes(bType)) {
    throw new Error('matmul inputs must be float32 or float16');
  }
  if ((aShape.length < 2) || (bShape.length < 2)) {
    throw new Error('matmul inputs must be at least 2D');
  }

  return `
    const ${outputVar} = builder.matmul(
      ${inputVars[0]},
      ${inputVars[1]}
    );`;
}