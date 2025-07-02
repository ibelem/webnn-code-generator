import { freeDimsOverrides, getModelState } from '../../ui'
import { getTypedArrayName } from '../../utils';

// Extract variable names from ONNX node inputs/outputs
export function getInputVars(node: any, toJsVarName: (name: string) => string): string[] {
  const result: string[] = [];
  
  // Handle both single-input case and array-of-inputs case (like Concat)
  if (node.inputs) {
    for (const input of node.inputs) {
      if (input.name === "inputs" && Array.isArray(input.value)) {
        // Special case for operations like Concat with multiple inputs
        for (const val of input.value) {
          if (val.name) {
            result.push(toJsVarName(getNonEmptyStringAroundNewline(val.name)));
          }
        }
      } else if (input.value?.[0]?.name) {
        // Regular case - single input
        result.push(toJsVarName(getNonEmptyStringAroundNewline(input.value[0].name)));
      }
    }
  }
  
  return result;
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
export function getShape(
  node: any,
  idx: number = 0,
  nhwc: boolean = false
): { shape: number[], info: { conv: boolean, depthwise: boolean, convTranspose: boolean } } {
  let shape = node.inputs?.[idx]?.value?.[0]?.type?.shape?.dimensions || [];
  shape = applyFreeDimsOverrides(shape, freeDimsOverrides);

  // Detect op type and depthwise regardless of layout
  const nodeType = node.type?.name?.toLowerCase() || '';
  const conv = nodeType.includes('conv');
  const convTranspose = nodeType.includes('convtranspose') || nodeType.includes('transposeconv');
  const groupsAttr = node.attributes?.find((a: any) => a.name === 'group' || a.name === 'groups');
  const groups = groupsAttr ? Number(Array.isArray(groupsAttr.value) ? groupsAttr.value[0] : groupsAttr.value) : 1;
  const inChannels = shape[1];
  const outChannels = shape[0];
  const depthwise = conv && groups === inChannels && (outChannels % inChannels === 0);

  // Only the returned shape is impacted by layout
  if (nhwc && shape.length === 4) {
    if (depthwise) {
      // Depthwise Conv: OIHW -> IHWO
      return { shape: [shape[1], shape[2], shape[3], shape[0]], info: { conv, depthwise, convTranspose } };
    }
    if (convTranspose) {
      // ConvTranspose: OIHW -> HWIO
      return { shape: [shape[2], shape[3], shape[1], shape[0]], info: { conv, depthwise, convTranspose } };
    }
    if (conv) {
      // Regular Conv: OIHW -> OHWI
      return { shape: [shape[0], shape[2], shape[3], shape[1]], info: { conv, depthwise, convTranspose } };
    }
    // Default: NCHW -> NHWC
    return { shape: [shape[0], shape[2], shape[3], shape[1]], info: { conv, depthwise, convTranspose } };
  }

  // For NCHW or other layouts, just return the original shape and correct info
  return { shape, info: { conv, depthwise, convTranspose } };
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

export function getInitializerArr(node: any, idx: number, nhwc: boolean): number[] | undefined {
    if (node.inputs.length > idx && node.inputs[idx]?.value?.[0]?.initializer) {
      const init = node.inputs[idx].value[0].initializer;
      let arr;
      if (init.values) {
        arr = Object.keys(init.values)
          .sort((a, b) => Number(a) - Number(b))
          .map(k => init.values[k]);
        return arr;
      } else {
          const { weightNchwBin, weightNhwcBin } = getModelState();
          const dataType = getDataType(node, idx);
          let weightsDataOffset = init.dataOffset;
          let weightsByteLength = init.byteLength;
          const weights_array_buffer = nhwc ? weightNhwcBin : weightNchwBin;
          if (!weights_array_buffer) {
            throw new Error('Weights array buffer is null');
          }
          // Use getTypedArrayName to select the correct typed array constructor
          const TypedArrayCtorName = getTypedArrayName(dataType);
          if (!TypedArrayCtorName || !(TypedArrayCtorName in window)) {
            throw new Error(`TypedArray constructor "${TypedArrayCtorName}" not found for dataType "${dataType}"`);
          }
          // @ts-ignore
          const TypedArrayCtor = window[TypedArrayCtorName];
          const js_shape_array = new TypedArrayCtor(weights_array_buffer, weightsDataOffset, weightsByteLength / TypedArrayCtor.BYTES_PER_ELEMENT);
          return Array.from(js_shape_array, Number);
      }
 
    }
    return undefined;
  }