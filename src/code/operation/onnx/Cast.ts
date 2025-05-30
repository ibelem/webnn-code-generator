import {
  getInputVars,
  getOutputVars
} from '../operation-utils';

/**
 * Generate JavaScript code for a WebNN cast operation from ONNX Cast node info.
 * https://www.w3.org/TR/webnn/#api-mlgraphbuilder-cast
 */
export function Cast(
  node: any,
  toJsVarName: (name: string) => string
): string {
  const inputVars = getInputVars(node, toJsVarName);
  const outputVars = getOutputVars(node, toJsVarName);

  // ONNX Cast type is usually in node.attributes[0].value or node.attributes.find(a => a.name === 'to')
  let toType = undefined;
  for (const attr of node.attributes || []) {
    if (attr.name === 'to') {
      // ONNX type is usually an integer enum, map to WebNN string type if needed
      // If already a string, use directly
      if (typeof attr.value === 'string') {
        toType = attr.value;
      } else if (typeof attr.value?.value === 'string') {
        toType = attr.value.value;
      } else if (typeof attr.value === 'number' || typeof attr.value?.value === 'number') {
        // Map ONNX type enum to WebNN string
        // Common ONNX type enums:
        // 1=float32, 2=uint8, 3=int8, 4=uint16, 5=int16, 6=int32, 7=int64, 9=bool, 10=float16, 11=double, 12=uint32, 13=uint64, 14=complex64, 15=complex128
        const typeMap: { [key: number]: string } = {
          1: 'float32',
          2: 'uint8',
          3: 'int8',
          4: 'uint16',
          5: 'int16',
          6: 'int32',
          7: 'int64',
          10: 'float16',
          11: 'float64',
          12: 'uint32',
          13: 'uint64'
        };
        const typeNum = typeof attr.value === 'number' ? attr.value : attr.value.value;
        toType = typeMap[typeNum] || 'float32';
      }
      break;
    }
  }

  if (!toType) {
    throw new Error('ONNX Cast node missing "to" attribute for target type.');
  }

  return `
    const ${outputVars[0]} = builder.cast(
      ${inputVars[0]},
      '${toType}'
    );`;
}