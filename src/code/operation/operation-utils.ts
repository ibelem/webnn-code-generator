// Extract variable names from ONNX node inputs/outputs
export function getInputVars(node: any, toJsVarName: (name: string) => string): string[] {
  return (node.inputs || [])
    .map((i: any) => getNonEmptyStringAroundNewline(i.value?.[0]?.name))
    .map(toJsVarName);
}

export function getOutputVars(node: any, toJsVarName: (name: string) => string): string[] {
  return (node.outputs || [])
    .map((o: any) => getNonEmptyStringAroundNewline(o.value?.[0]?.name))
    .map(toJsVarName);
}

// Extract shape and dtype from a node input/output
export function getShape(node: any, idx: number = 0): number[] {
  return node.inputs?.[idx]?.value?.[0]?.type?.shape?.dimensions || [];
}

export function getDtype(node: any, idx: number = 0): string {
  return node.inputs?.[idx]?.value?.[0]?.type?.dataType || '';
}

// Centralized dtype/rank validation
export function validateDtype(dtype: string, allowed: string[], op: string) {
  if (!allowed.includes(dtype)) {
    throw new Error(`${op}: dtype must be one of ${allowed.join(', ')}, got ${dtype}`);
  }
}

export function validateRank(shape: number[], minRank: number, op: string) {
  if (shape.length < minRank) {
    throw new Error(`${op}: rank must be >= ${minRank}, got ${shape.length}`);
  }
}

// Utility for inlined reshape expression
export function inlineReshape(varName: string, fromShape: number[], toShape: number[]): string {
  if (JSON.stringify(fromShape) === JSON.stringify(toShape)) return varName;
  return `builder.reshape(${varName}, [${toShape.join(', ')}])`;
}

// Utility for zero constant
export function zeroConstant(dtype: string, shape: number[]): string {
  const elemCount = shape.reduce((a, b) => a * (typeof b === 'number' ? b : 1), 1) || 1;
  const typedArrayCtor =
    dtype === 'uint8' ? 'Uint8Array' :
    dtype === 'int8' ? 'Int8Array' :
    dtype === 'uint32' ? 'Uint32Array' :
    'Int32Array';
  return `builder.constant(
    {dataType: '${dtype}', shape: [${shape.join(', ')}]},
    new ${typedArrayCtor}([${Array(elemCount).fill(0).join(', ')}])
  )`;
}

// You may need to import getNonEmptyStringAroundNewline from your utils
import { getNonEmptyStringAroundNewline } from '../../utils';