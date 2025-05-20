import { getModelState } from '../../ui';

/**
 * Generate JavaScript code for a WebNN reshape operation from ONNX Reshape node info.
 * @param node - The ONNX node object (with inputs, outputs)
 * @param toJsVarName - Function to convert ONNX names to JS variable names
 * @returns JavaScript code string for the reshape operation
 */
export function reshape_js(
  node: any,
  toJsVarName: (name: string) => string
): string {
  // Get input and output variable names
  const inputs: string[] = node.inputs?.map((i: any) => i.value?.[0]?.name) || [];
  const outputs: string[] = node.outputs?.map((o: any) => o.value?.[0]?.name) || [];
  const inputVars = inputs.map(toJsVarName);
  const outputVar = toJsVarName(outputs[0]);

  // Find the shape input and its initializer
  const shapeInput = node.inputs?.[1];
  if (!shapeInput || !Array.isArray(shapeInput.value) || !shapeInput.value[0]) {
    throw new Error('Reshape node missing shape input');
  }
  const shapeValue = shapeInput.value[0];
  const shapeName = shapeValue.name;
  const shapeInit = shapeValue.initializer;
  if (!shapeInit) {
    throw new Error(`Reshape shape input '${shapeName}' must be an initializer`);
  }

  const { weightModelData } = getModelState();
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
    const ${outputVar} = builder.reshape(
      ${inputVars[0]},
      ${js_shape}
    );`;
}