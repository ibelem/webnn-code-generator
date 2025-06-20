import {
  getOutputVars,
  getAttrValue
} from '../../operation-utils';

/**
 * Generate JavaScript code for a WebNN constant operation from ONNX Constant node info.
 * https://www.w3.org/TR/webnn/#api-mlgraphbuilder-constant
 */
export function Constant(
  node: any,
  toJsVarName: (name: string) => string,
  options: { [key: string]: any } = {}
): string {
  const nhwc = !!options.nhwc;
  const outputVars = getOutputVars(node, toJsVarName);

  // ONNX Constant can have value_float, value_int, value_floats, value_ints, value_tensor, etc.
  // We'll handle value_tensor (most common), and scalars/arrays.

  // Try to get value_tensor first
  const valueTensor = getAttrValue(node, 'value', undefined);

  if (valueTensor && valueTensor.type && valueTensor.type.shape && valueTensor.type.dataType && valueTensor.values) {
    // value_tensor: {type: {dataType, shape: {dimensions}}, values: {...}}
    const dtype = valueTensor.type.dataType;
    const shape = valueTensor.type.shape.dimensions;
    // ONNX stores values as { "0": v0, "1": v1, ... }
    const valuesArr = Array.isArray(valueTensor.values)
      ? valueTensor.values
      : Object.values(valueTensor.values);
    const typedArrayCtor =
      dtype === 'float32' ? 'Float32Array' :
      dtype === 'int64' ? 'BigInt64Array' :
      dtype === 'float16' ? 'Float16Array' :
      dtype === 'float64' ? 'Float64Array' :
      dtype === 'int32' ? 'Int32Array' :
      dtype === 'uint64' ? 'BigUint64Array' :
      dtype === 'int8' ? 'Int8Array' :
      dtype === 'uint8' ? 'Uint8Array' :
      dtype === 'int16' ? 'Int16Array' :
      dtype === 'uint16' ? 'Uint16Array' :
      dtype === 'uint32' ? 'Uint32Array' :
      dtype === 'bool' ? 'Uint8Array' :
      'Float32Array'; // fallback

    // Only permute shapes for 4D constants that might be convolution weights
    if (nhwc && shape.length === 4) {
      // For conv weights: OIHW -> IHWO
      const permutedShape = [shape[1], shape[2], shape[3], shape[0]];

      // Permute the data from OIHW to IHWO
      function permuteOIHWtoIHWO(data: number[], shape: number[]): number[] {
        const [O, I, H, W] = shape;
        const result = new Array(I * H * W * O);
        let idx = 0;
        for (let i = 0; i < I; ++i)
          for (let h = 0; h < H; ++h)
            for (let w = 0; w < W; ++w)
              for (let o = 0; o < O; ++o)
                result[idx++] = data[
                  ((o * I + i) * H + h) * W + w
                ];
        return result;
      }

      const permutedData = permuteOIHWtoIHWO(valuesArr, shape);

      return `
        // Original shape: [${shape.join(', ')}], transformed to NHWC (IHWO): [${permutedShape.join(', ')}]
        const ${outputVars[0]} = builder.constant(
          {type: '${dtype}', shape: [${permutedShape.join(', ')}]},
          new ${typedArrayCtor}([${permutedData.join(', ')}])
        );
      `;
    }

    return `
      const ${outputVars[0]} = builder.constant(
        {type: '${dtype}', shape: [${shape.join(', ')}]},
        new ${typedArrayCtor}([${valuesArr.join(', ')}])
      );
    `;
  }

  // Handle scalar float/int
  const valueFloat = getAttrValue(node, 'value_float', undefined);
  if (valueFloat !== undefined) {
    return `
      const ${outputVars[0]} = builder.constant('float32', ${valueFloat});
    `;
  }
  const valueInt = getAttrValue(node, 'value_int', undefined);
  if (valueInt !== undefined) {
    return `
      const ${outputVars[0]} = builder.constant('int32', ${valueInt});
    `;
  }

  // Handle value_floats/value_ints (1D arrays)
  const valueFloats = getAttrValue(node, 'value_floats', undefined);
  if (valueFloats !== undefined && Array.isArray(valueFloats)) {
    return `
      const ${outputVars[0]} = builder.constant(
        {dataType: 'float32', shape: [${valueFloats.length}]},
        new Float32Array([${valueFloats.join(', ')}])
      );
    `;
  }
  const valueInts = getAttrValue(node, 'value_ints', undefined);
  if (valueInts !== undefined && Array.isArray(valueInts)) {
    return `
      const ${outputVars[0]} = builder.constant(
        {dataType: 'int32', shape: [${valueInts.length}]},
        new Int32Array([${valueInts.join(', ')}])
      );
    `;
  }

  // Fallback: error
    return `// ERROR: Unsupported Constant node format for ${node.name || outputVars[0]}`;
  }