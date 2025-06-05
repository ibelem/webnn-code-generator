import { getModelState } from './ui';

export function toJsVarName(name: string): string {
  // Prefix with 'var_' if not a valid identifier or starts with a digit
  let jsVarName = name;
  // Check if the name is a valid JS identifier and does not start with a digit
  if (!/^[$A-Z_a-z][$\w]*$/.test(jsVarName) || /^\d/.test(jsVarName)) {
    jsVarName = `var_${jsVarName}`;
  }
  // Replace any character that is not alphanumeric or underscore with '_'
  jsVarName = jsVarName.replace(/[^a-zA-Z0-9_]/g, '_');

  // Replace the string to lowercase
  // This is to ensure that the variable name is consistent with the ONNX naming convention
  // and to avoid any potential conflicts with reserved keywords in JavaScript.
  jsVarName = jsVarName.toLowerCase();
  return jsVarName;
}

/**
 * TFLite model
 * There are some cases where the string contains '\n' characters,
 * but we want to ignore them. 
 * Returns the substring before the first '\n' character.
 * If that substring is empty, returns the substring after the first '\n' character.
 * If '\n' is not found, returns the original string.
 */
export function getNonEmptyStringAroundNewline(str: string): string {
  if (typeof str !== 'string') return str;
  const idx = str.indexOf('\n');
  if (idx === -1) return str;
  const before = str.substring(0, idx).trim();
  if (before.length > 0) return before;
  // Return the substring after the first '\n', trimmed
  return str.substring(idx + 1).trim();
}

/**
 * TFLite model
 * Find a weight node in weightModelData by its 'name' property.
 * @param weightModelData - The weights object (e.g. from weights.json)
 * @param name - The name to match (e.g. "conv2d/Kernel")
 * @returns The node value if found, otherwise undefined
 */
export function findWeightNodeByName(weightModelData: Record<string, any> | null, name: string) {
  if (!weightModelData || !name) return undefined;
  return Object.values(weightModelData).find((node: any) => node && node.name === name);
}

/**
 * Given a data type string (e.g. "float32", "int64"), return the corresponding TypedArray constructor name as a string.
 * Example: "float32" -> "Float32Array", "int64" -> "BigInt64Array"
 */
export function getTypedArrayName(dataType: string): string | undefined {
  switch (dataType) {
    case 'float32': return 'Float32Array';
    case 'uint8': return 'Uint8Array';
    case 'int8': return 'Int8Array';
    case 'uint16': return 'Uint16Array';
    case 'int16': return 'Int16Array';
    case 'int32': return 'Int32Array';
    case 'int64': return 'BigInt64Array';
    case 'bool': return 'Uint8Array'; // BOOL is treated as Uint8Array
    case 'float16': return 'Float16Array';
    case 'float64': return 'Float64Array';
    case 'uint32': return 'Uint32Array';
    case 'uint64': return 'BigUint64Array';
    default: return 'Float32Array';
  }
}

export function getDatafromWeightsArrayBuffer(
  arrayBuffer: ArrayBuffer,
  dataType: string,
  dataOffset: number,
  byteLength: number
): any {
  // Create a Uint8Array view for the slice
  const buffer = new Uint8Array(arrayBuffer, dataOffset, byteLength);
  const baseOffset = buffer.byteOffset;
  const baseBuffer = buffer.buffer;
  let data;
  switch (dataType.toLowerCase()) {
    case 'float':
    case 'float32':
      data = new Float32Array(baseBuffer, baseOffset, byteLength / Float32Array.BYTES_PER_ELEMENT);
      break;
    case 'uint8':
      data = new Uint8Array(baseBuffer, baseOffset, byteLength);
      break;
    case 'int8':
      data = new Int8Array(baseBuffer, baseOffset, byteLength);
      break;
    case 'uint16':
      data = new Uint16Array(baseBuffer, baseOffset, byteLength / Uint16Array.BYTES_PER_ELEMENT);
      break;
    case 'int16':
      data = new Int16Array(baseBuffer, baseOffset, byteLength / Int16Array.BYTES_PER_ELEMENT);
      break;
    case 'int32':
      data = new Int32Array(baseBuffer, baseOffset, byteLength / Int32Array.BYTES_PER_ELEMENT);
      break;
    case 'int64':
      if (typeof BigInt64Array !== 'undefined') {
        if (baseOffset % 8 !== 0) throw new Error('BigInt64Array offset must be a multiple of 8');
        data = new BigInt64Array(baseBuffer, baseOffset, byteLength / BigInt64Array.BYTES_PER_ELEMENT);
      } else {
        data = "Int64 data type not fully supported in this browser";
      }
      break;
    case 'float16':
      // Not standard; handle or polyfill as needed
      data = "Float16Array not supported natively";
      break;
    case 'double':
    case 'float64':
      data = new Float64Array(baseBuffer, baseOffset, byteLength / Float64Array.BYTES_PER_ELEMENT);
      break;
    case 'uint32':
      data = new Uint32Array(baseBuffer, baseOffset, byteLength / Uint32Array.BYTES_PER_ELEMENT);
      break;
    case 'uint64':
      if (typeof BigUint64Array !== 'undefined') {
        if (baseOffset % 8 !== 0) throw new Error('BigUint64Array offset must be a multiple of 8');
        data = new BigUint64Array(baseBuffer, baseOffset, byteLength / BigUint64Array.BYTES_PER_ELEMENT);
      } else {
        data = "Uint64 data type not fully supported in this browser";
      }
      break;
    case 'string':
      data = new TextDecoder().decode(new Uint8Array(baseBuffer, baseOffset, byteLength));
      break;
    case 'bool':
      data = Array.from(new Uint8Array(baseBuffer, baseOffset, byteLength)).map(Boolean);
      break;
    default:
      data = "Unknown data type: " + dataType;
  }
  return data;
}

// https://www.w3.org/TR/webnn/#enumdef-mloperanddatatype
export function mlOperandDataType(onnxType: string): string {
  switch (onnxType.toLowerCase()) {
    case 'float32': return 'float32';
    case 'float16': return 'float16';
    case 'int32':   return 'int32';
    case 'uint32':  return 'uint32';
    case 'int64':   return 'int64';
    case 'uint64':  return 'uint64';
    case 'int8':    return 'int8';
    case 'uint8':   return 'uint8';
    default:        return 'int32'; // fallback to int32
  }
}

export function getWeightInfo(name: string, weightModelData:any) {
  if (!weightModelData || !weightModelData[name]) return null;
  const w = weightModelData[name];
  return {
    dataOffset: w.dataOffset,
    byteLength: w.byteLength,
    dataType: w.dataType,
    shape: w.shape
  };
}

export function modelName() {
  const { graphModelData } = getModelState();
  if (!graphModelData) return null;
  let modelName = graphModelData?.identifier || 'WebNN';
  if (graphModelData.name) {
    modelName = graphModelData.name;
  }
  modelName = modelName.replaceAll('.tflite', '').replaceAll('.onnx', '');

  // Replace "-" with "_" for consistent camelCase conversion
  modelName = modelName.replace(/-/g, '_');

  // Convert to camelCase: remove "_" and capitalize the first letter after each "_"
  modelName = modelName.replace(/_([a-zA-Z0-9])/g, (_: string, c: string) => c.toUpperCase());

  // Capitalize the first character
  modelName = modelName.charAt(0).toUpperCase() + modelName.slice(1);

  return modelName;
}

export function hasKeysandNumberValues(obj: any): boolean {
  return (
    obj &&
    typeof obj === 'object' &&
    !Array.isArray(obj) &&
    Object.keys(obj).length > 0 &&
    Object.values(obj).every(
      (value) =>
        typeof value === 'number' &&
        isFinite(value) &&
        !isNaN(value)
    )
  );
}

export function downloadFile(name: string, type: string, code: string) {
  const blob = new Blob([code], { type: type });
  const url = URL.createObjectURL(blob);

  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}