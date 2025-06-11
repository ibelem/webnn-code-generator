import {
  getOutputVars
} from '../../operation-utils';

/**
 * Generate JavaScript code for a WebNN constant operand.
 * https://www.w3.org/TR/webnn/#api-mlgraphbuilder-constant
 */
export function Constant(
  node: any,
  toJsVarName: (name: string) => string,
  options: { nhwc?: boolean } = {}
): string {
  // Get output variable name
  const outputVars = getOutputVars(node, toJsVarName);
  const outputVar = outputVars[0];

  // Find the 'value' attribute
  const valueAttr = (node.attributes || []).find((attr: any) => attr.name === 'value' && attr.t);
  if (!valueAttr) {
    return `// Constant node ${outputVar} missing value tensor.`;
  }

  // Todo
  const t = valueAttr.t;

  // Map ONNX/WebNN dataType to JS typed array and WebNN dtype
  const typeMap: Record<number | string, { js: string; webnn: string }> = {
    1: { js: 'Float32Array', webnn: 'float32' },
    2: { js: 'Uint8Array', webnn: 'uint8' },
    3: { js: 'Int8Array', webnn: 'int8' },
    4: { js: 'Uint16Array', webnn: 'uint16' },
    5: { js: 'Int16Array', webnn: 'int16' },
    6: { js: 'Int32Array', webnn: 'int32' },
    7: { js: 'BigInt64Array', webnn: 'int64' },
    9: { js: 'Uint8Array', webnn: 'bool' },
    10: { js: 'Float16Array', webnn: 'float16' },
    11: { js: 'Float64Array', webnn: 'float64' },
    12: { js: 'Uint32Array', webnn: 'uint32' },
    13: { js: 'BigUint64Array', webnn: 'uint64' },
  };

  const dataType = t.dataType ?? 1;
  const { js: typedArray, webnn: webnn_dtype } = typeMap[dataType] || { js: 'Float32Array', webnn: 'float32' };
  let shape = Array.isArray(t.dims) ? t.dims.map(Number) : [];

  // Find the data array
  let js_data = '';
  if (t.floatData) {
    js_data = `[${t.floatData.join(', ')}]`;
  } else if (t.uint8Data) {
    js_data = `[${t.uint8Data.join(', ')}]`;
  } else if (t.int8Data) {
    js_data = `[${t.int8Data.join(', ')}]`;
  } else if (t.int16Data) {
    js_data = `[${t.int16Data.join(', ')}]`;
  } else if (t.int32Data) {
    js_data = `[${t.int32Data.join(', ')}]`;
  } else if (t.int64Data) {
    js_data = `[${t.int64Data.join(', ')}]`;
  } else if (t.boolData) {
    js_data = `[${t.boolData.join(', ')}]`;
  } else if (t.float16Data) {
    js_data = `[${t.float16Data.join(', ')}]`;
  } else if (t.float64Data) {
    js_data = `[${t.float64Data.join(', ')}]`;
  } else if (t.uint32Data) {
    js_data = `[${t.uint32Data.join(', ')}]`;
  } else if (t.uint64Data) {
    js_data = `[${t.uint64Data.join(', ')}]`;
  } else {
    // fallback: fill with zeros
    const size = shape.reduce((a: number, b: number) => a * b, 1) || 1;
    js_data = `[${Array(size).fill(0).join(', ')}]`;
  }

  const nhwc = !!options.nhwc;
  
  // Only permute shapes for 4D constants that might be convolution weights
  if (nhwc && shape.length === 4) {
    // For conv weights: OIHW -> OHWI
    shape = [shape[0], shape[2], shape[3], shape[1]];
    
    // Add comment to indicate shape transformation
    return `
    // Original shape: [${t.dims.join(', ')}], transformed to NHWC: [${shape.join(', ')}]
    const ${outputVar} = builder.constant(
      { dataType: '${webnn_dtype}', shape: [${shape.join(', ')}] },
      new ${typedArray}(${js_data})
    );`;
  }
  
  // Regular constant
  return `
    const ${outputVar} = builder.constant(
      { dataType: '${webnn_dtype}', shape: [${shape.join(', ')}] },
      new ${typedArray}(${js_data})
    );`;
}