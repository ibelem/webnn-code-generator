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

export const dataTypeMap = {
  1: 'float32',   // FLOAT
  2: 'uint8',     // UINT8
  3: 'int8',      // INT8
  4: 'uint16',    // UINT16
  5: 'int16',     // INT16
  6: 'int32',     // INT32
  7: 'int64',     // INT64 (not directly supported in JS)
  9: 'uint8',     // BOOL (special handling)
  10: 'float16',  // FLOAT16 (not directly supported in JS)
  11: 'float64',  // DOUBLE
  12: 'uint32',   // UINT32
  13: 'uint64',   // UINT64 (not directly supported in JS)
}

export const typedArrayMap = {
  1: 'Float32Array',   // FLOAT
  2: 'Uint8Array',     // UINT8
  3: 'Int8Array',      // INT8
  4: 'Uint16Array',    // UINT16
  5: 'Int16Array',     // INT16
  6: 'Int32Array',     // INT32
  7: 'BigInt64Array',  // INT64 (not directly supported in JS)
  9: 'Bool',           // BOOL (special handling)
  10: 'Float16Array',  // FLOAT16 (not directly supported in JS)
  11: 'Float64Array',  // DOUBLE
  12: 'Uint32Array',   // UINT32
  13: 'BigUint64Array',   // UINT64 (not directly supported in JS)
}

/**
 * Get the ONNX data type code from a string (e.g., "float32" -> 1, "int64" -> 7)
 */
export function getDataTypeCode(typeStr: string): number | undefined {
  for (const [key, value] of Object.entries(dataTypeMap)) {
    if (value === typeStr) {
      return Number(key);
    }
  }
  return undefined;
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
    case 'float64': return 'Float64Array';
    case 'uint32': return 'Uint32Array';
    case 'uint64': return 'BigUint64Array';
    case 'bool': return 'Uint8Array'; // BOOL is treated as Uint8Array
    case 'float16': return 'Float16Array';
    default: return 'Float32Array';
  }
}

export function weightInfo(name: string) {
  const { weightModelData } = getModelState();
  if (!weightModelData || !weightModelData[name]) return null;
  const w = weightModelData[name];
  return {
    dataOffset: w.dataOffset,
    byteLength: w.byteLength,
    dataType: w.dataType,
    shape: w.shape
  };
}

export function weightBuffer(name: string) {
  const { weightModelData, binaryModelData } = getModelState();
  if (!weightModelData || !binaryModelData || !weightModelData[name]) return null;
  const { dataOffset, byteLength, dataType } = weightModelData[name];
  const buffer = binaryModelData.slice(dataOffset, dataOffset + byteLength);

  switch (dataType) {
    case 'float32':
      return new Float32Array(buffer);
    case 'uint8':
      return new Uint8Array(buffer);
    case 'int8':
      return new Int8Array(buffer);
    case 'uint16':
      return new Uint16Array(buffer);
    case 'int16':
      return new Int16Array(buffer);
    case 'int32':
      return new Int32Array(buffer);
    case 'int64':
      return new BigInt64Array(buffer);
    case 'bool':
      return new Uint8Array(buffer); // BOOL is treated as UINT8
    case 'float16':
      // @ts-ignore
      return new Float16Array(buffer);
    case 'float64':
      return new Float64Array(buffer);
    case 'uint32':
      return new Uint32Array(buffer);
    case 'uint64':
      return new BigUint64Array(buffer);
    default:
      return new Uint8Array(buffer);
  }
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