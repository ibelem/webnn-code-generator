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

export function applyFreeDimsOverrides(shape: (string|number)[], freeDimsOverrides: Record<string, number | null>): (string|number)[] {
  return shape.map(dim => {
    if (typeof dim === 'string' && freeDimsOverrides.hasOwnProperty(dim)) {
      const override = freeDimsOverrides[dim];
      return override !== null ? override : dim;
    }
    return dim;
  });
}

// Update getAttrValue to take node, not attrs
export function getAttrValue(node: any, name: string, defaultValue: any = undefined): any {
  const attrs = node?.attributes || [];
  const attr = attrs.find((a: any) => a.name === name);
  if (!attr) return defaultValue;
  if (attr.value && typeof attr.value === 'object' && 'value' in attr.value) {
    // Handle bigint or bigint[]
    if (Array.isArray(attr.value.value)) {
      return attr.value.value.map((v: any) => Number(v));
    }
    return Number(attr.value.value);
  }
  // Fallback for direct value
  return typeof attr.value !== 'undefined' ? attr.value : defaultValue;
}

// Extract shape and dtype from a node input/output
export function getShape(node: any, idx: number = 0, nhwc: boolean = false): number[] {
  let shape = node.inputs?.[idx]?.value?.[0]?.type?.shape?.dimensions || [];
  shape = applyFreeDimsOverrides(shape, freeDimsOverrides);

  // Align with https://github.com/ibelem/netron/blob/webnn-netron-2/source/view.js#L1108
  if (nhwc && shape.length === 4) {
    const nodeType = node.type?.name?.toLowerCase() || '';
    const isConv = nodeType.includes('conv');
    const isConvTranspose = nodeType.includes('convtranspose') || nodeType.includes('transposeconv');
    const groupsAttr = node.attributes?.find((a: any) => a.name === 'group' || a.name === 'groups');
    const groups = groupsAttr ? Number(Array.isArray(groupsAttr.value) ? groupsAttr.value[0] : groupsAttr.value) : 1;
    const inChannels = shape[1];
    const outChannels = shape[0];
    const isDepthwise = isConv && groups === inChannels && (outChannels % inChannels === 0);

    if (isDepthwise) {
      // Depthwise Conv: OIHW -> IHWO
      return [shape[1], shape[2], shape[3], shape[0]];
    }
    if (isConvTranspose) {
      // ConvTranspose: OIHW -> HWIO (common for TF, check your framework)
      return [shape[2], shape[3], shape[1], shape[0]];
    }
    if (isConv) {
      // Regular Conv: OIHW -> OHWI
      return [shape[0], shape[2], shape[3], shape[1]];
    }
    // Default: NCHW -> NHWC
    return [shape[0], shape[2], shape[3], shape[1]];
  }

  return shape;
}

export function getDataType(node: any, idx: number = 0): string {
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