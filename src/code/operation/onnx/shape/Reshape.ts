import {
  getInputVars,
  getOutputVars
} from '../../operation-utils';
import { getModelState } from '../../../../ui';
import { extractTensorMetadataFromGraphJson } from '../../../../utils';

/**
 * Generate JavaScript code for a WebNN reshape operation from ONNX Reshape node info.
 * https://www.w3.org/TR/webnn/#api-mlgraphbuilder-reshape-method
 * https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/webnn/builders/impl/reshape_op_builder.cc
 */
export function Reshape(
  node: any,
  toJsVarName: (name: string) => string,
  options: { [key: string]: any } = {}
): string {
  const nhwc = !!options.nhwc;
  const { graphModelData, weightNchwBin, weightNhwcBin } = getModelState();
  const tensorMap = extractTensorMetadataFromGraphJson(graphModelData);
  const inputVars = getInputVars(node, toJsVarName);
  const outputVars = getOutputVars(node, toJsVarName);

  // Try ONNX style: shape input as tensor
  const shapeInput = node.inputs?.[1];
  if (shapeInput && Array.isArray(shapeInput.value) && shapeInput.value[0]) {
    const shapeValue = shapeInput.value[0];
    const shapeName = shapeValue.name;
    const shapeInfo = tensorMap[shapeName];
    if (!shapeInfo) {
      throw new Error(`Reshape shape initializer '${shapeName}' not found in graphModelData`);
    }
    const shape_offset = shapeInfo.dataOffset;
    const shape_length = shapeInfo.byteLength;
    if (shape_offset == null) {
      throw new Error(`Reshape shape initializer '${shapeName}' missing offset`);
    }
    if (shape_length == null) {
      throw new Error(`Reshape shape initializer '${shapeName}' missing length`);
    }
    
    // Only support BigInt64Array for shape tensor
    const weights_array_buffer = nhwc ? weightNhwcBin : weightNchwBin;
    if (!weights_array_buffer) {
      throw new Error('Weights array buffer is null');
    }
    const js_shape_array = new BigInt64Array(weights_array_buffer, shape_offset, shape_length / BigInt64Array.BYTES_PER_ELEMENT);
    const array = Array.from(js_shape_array, Number);

    // Convert BigInt64Array to Number array for WebNN and handle -1
    const js_shape = `(() => {
        const shape = [${array}];
        // WebNN does not support 0 as a reshape dimension if allowzero is set
        const allowzero = ${node.attributes?.find((attr: any) => attr.name === 'allowzero')?.value === 1 ? 'true' : 'false'};
        if (allowzero && shape.some(v => v === 0)) {
          throw new Error('WebNN reshape does not support 0 as a dimension when allowzero is enabled');
        }
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

    // Add label option
    const labelOpt = node.name ? `{ label: '${node.name}' }` : '{}';

    return `
    const ${outputVars[0]} = builder.reshape(
      ${inputVars[0]},
      ${js_shape},
      ${labelOpt}
    );`;
  }

  // TFLite style: use new_shape attribute
  const newShapeAttr = node.attributes?.find((attr: any) => attr.name === 'new_shape');
  if (newShapeAttr && Array.isArray(newShapeAttr.value)) {
    const shapeArr = newShapeAttr.value.map((v: any) => Number(v));
    const labelOpt = node.name ? `{ label: '${node.name}' }` : '{}';
    return `
    const ${outputVars[0]} = builder.reshape(
      ${inputVars[0]},
      [${shapeArr.join(', ')}],
      ${labelOpt}
    );`;
  }

  throw new Error('Reshape node missing shape input or new_shape attribute');
}