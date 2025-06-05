import {
  getInputVars,
  getOutputVars
} from '../../operation-utils';

/**
 * Generate JavaScript code for a WebNN reshape operation from ONNX Reshape node info.
 * https://www.w3.org/TR/webnn/#api-mlgraphbuilder-reshape-method
 */
export function Reshape(
  node: any,
  toJsVarName: (name: string) => string,
  options: { nhwc?: boolean, weightModelData?: Record<string, any> } = {}
): string {
  const inputVars = getInputVars(node, toJsVarName);
  const outputVars = getOutputVars(node, toJsVarName);

  // Try ONNX style: shape input as tensor
  const shapeInput = node.inputs?.[1];
  if (shapeInput && Array.isArray(shapeInput.value) && shapeInput.value[0]) {
    const shapeValue = shapeInput.value[0];
    const shapeName = shapeValue.name;
    const weightModelData = options.weightModelData;
    if (!weightModelData) {
      throw new Error('Weight model data is not available');
    }
    let shape_offset: number | null = null;
    let shape_length: number | null = null;
    shape_offset = weightModelData?.[shapeName]?.dataOffset;
    shape_length = weightModelData?.[shapeName]?.byteLength;
    if (shape_offset === null) {
      throw new Error(`Reshape shape initializer '${shapeName}' missing offset`);
    }
    if (shape_length === null) {
      throw new Error(`Reshape shape initializer '${shapeName}' missing length`);
    }
    // Only support BigInt64Array for shape tensor
    const js_shape_array = `new BigInt64Array(weights_array_buffer, ${shape_offset}, ${shape_length} / BigInt64Array.BYTES_PER_ELEMENT)`;

    // Convert BigInt64Array to Number array for WebNN and handle -1
    const js_shape = `(() => {
        const shape = Array.from(${js_shape_array}, Number);
        // Calculate the concrete size for value -1.
        if (shape.includes(-1)) {
          const count = shape.filter(v => v === -1).length;
          if (count !== 1) {
            throw new Error('Only one -1 is allowed in reshape shape');
          }
          const totalInput = ${inputVars[0]}.shape.reduce((a, b) => a * b, 1);
          const known = shape.reduce((a, b) => b === -1 ? a : a * b, 1);
          const idx = shape.indexOf(-1);
          shape[idx] = totalInput / known;
        }
        return shape;
      })()`;

    return `
    const ${outputVars[0]} = builder.reshape(
      ${inputVars[0]},
      ${js_shape}
    );`;
  }

  // TFLite style: use new_shape attribute
  const newShapeAttr = node.attributes?.find((attr: any) => attr.name === 'new_shape');
  if (newShapeAttr && Array.isArray(newShapeAttr.value)) {
    const shapeArr = newShapeAttr.value.map((v: any) => Number(v));
    return `
      const ${outputVars[0]} = builder.reshape(
        ${inputVars[0]},
        [${shapeArr.join(', ')}]
      );`;
  }

  throw new Error('Reshape node missing shape input or new_shape attribute');
}