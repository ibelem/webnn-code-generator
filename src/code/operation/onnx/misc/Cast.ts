import {
  getInputVars,
  getOutputVars
} from '../../operation-utils';

/**
 * Generate JavaScript code for a WebNN cast operation from ONNX Cast node info.
 * https://www.w3.org/TR/webnn/#api-mlgraphbuilder-cast
 * https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/webnn/builders/impl/cast_op_builder.cc
 */
export function Cast(
  node: any,
  toJsVarName: (name: string) => string
): string {
  const inputVars = getInputVars(node, toJsVarName);
  const outputVars = getOutputVars(node, toJsVarName);

  // ONNX type enum to WebNN string type mapping
  // Common ONNX type enums:
  // 1=float32, 2=uint8, 3=int8, 4=uint16, 5=int16, 6=int32, 7=int64,
  // 9=bool, 10=float16, 11=double, 12=uint32, 13=uint64, 14=complex64, 15=complex128
  const typeMap: { [key: number]: string } = {
    0: 'undefined',
    1: 'float32',
    2: 'uint8',
    3: 'int8',
    4: 'uint16',
    5: 'int16',
    6: 'int32',
    7: 'int64',
    9: 'bool',
    10: 'float16',
    11: 'float64',
    12: 'uint32',
    13: 'uint64',
    14: 'complex64',
    15: 'complex128',
    16: 'bfloat16',
    17: 'int4',
    18: 'uint4'
  };

  let toType: string | undefined;
  for (const attr of node.attributes || []) {
    if (attr.name === 'to') {
      if (typeof attr.value === 'string') {
        toType = attr.value;
      } else if (typeof attr.value?.value === 'string') {
        toType = attr.value.value;
      } else if (typeof attr.value === 'number' || typeof attr.value?.value === 'number') {
        const typeNum = typeof attr.value === 'number' ? attr.value : attr.value.value;
        // Handle int64 support fallback
        if (typeNum === 7) {
          toType = 'int32';
        } else {
          toType = typeMap[typeNum] || 'float32';
        }
      }
      break;
    }
  }

  if (!toType) {
    throw new Error('ONNX Cast node missing "to" attribute for target type.');
  }

  // Add label option if node.name is present
  const labelOpt = node.name ? `{ label: '${node.name}' }` : '';

  return `
    const ${outputVars[0]} = builder.cast(
      ${inputVars[0]},
      '${toType}',
      ${labelOpt}
    );`;
}