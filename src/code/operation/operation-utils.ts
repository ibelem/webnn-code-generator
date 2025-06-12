import { freeDimsOverrides } from '../../ui'

export function permuteWeightShape(
  shape: number[],
  nhwc: boolean,
  nodeType: string,
  isDepthwise: boolean
): number[] {
  if (!nhwc || shape.length !== 4) return shape;

  // Depthwise Conv or ConvTranspose: OIHW -> IHWO (perm [1,2,3,0])
  if ((nodeType === 'Conv' && isDepthwise) || nodeType === 'ConvTranspose') {
    return [shape[1], shape[2], shape[3], shape[0]];
  }

  // Regular Conv: OIHW -> OHWI (perm [0,2,3,1])
  if (nodeType === 'Conv') {
    return [shape[0], shape[2], shape[3], shape[1]];
  }

  // Default: no permutation
  return shape;
}

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

function applyFreeDimsOverrides(shape: (string|number)[], freeDimsOverrides: Record<string, number | null>): (string|number)[] {
  return shape.map(dim => {
    if (typeof dim === 'string' && freeDimsOverrides.hasOwnProperty(dim)) {
      const override = freeDimsOverrides[dim];
      return override !== null ? override : dim;
    }
    return dim;
  });
}

// Extract shape and dtype from a node input/output
export function getShape(node: any, idx: number = 0, nhwc: boolean = false): number[] {
  let shape = node.inputs?.[idx]?.value?.[0]?.type?.shape?.dimensions || [];
  shape = applyFreeDimsOverrides(shape, freeDimsOverrides);

  // For 4D tensors, conditionally permute from NCHW to NHWC if needed
  if (nhwc && shape.length === 4) {
    const nodeType = node.type?.name;
    
    // For weights in ConvTranspose, we need special permutation
    if (idx === 1 && nodeType === 'ConvTranspose') {
      return [shape[0], shape[2], shape[3], shape[1]]; // OIHW -> OHWI
    }
    // For regular 4D tensors, do standard NCHW -> NHWC permutation
    return [shape[0], shape[2], shape[3], shape[1]]; // NCHW -> NHWC
  }
  
  return shape;
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

export function getAttr(node: any, name: string, defaultValue: any): any {
  if (!node || !node.attributes) return defaultValue;
  const attr = node.attributes.find((a: any) => a.name === name);
  if (attr === undefined) return defaultValue;
  // Try attr.value, then attr.value.value, else default
  if (attr.value !== undefined) {
    if (attr.value.value !== undefined) {
      return attr.value.value;
    }
    return attr.value;
  }
  return defaultValue;
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