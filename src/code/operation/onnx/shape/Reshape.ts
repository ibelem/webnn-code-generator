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
  
  // Common code for both ONNX and TFLite shape handling
  const generateShapeProcessingCode = (shapeArray: any) => {
    return `(() => {
      // If newShape's size is 0, then set outputShape to an empty list for a scalar
      const initialShape = [${shapeArray}];
      if (initialShape.length === 0) {
        return [];
      }
      
      let shape = [...initialShape];
      
      // Handle 0 dimensions (copy from input shape)
      const inputShape = ${inputVars[0]}.shape;
      for (let i = 0; i < shape.length; i++) {
        if (shape[i] === 0 && i < inputShape.length) {
          shape[i] = inputShape[i];
        }
      }
      
      // Calculate the concrete size for value -1
      if (shape.includes(-1)) {
        const count = shape.filter(v => v === -1).length;
        if (count !== 1) {
          throw new TypeError('Only one -1 is allowed in reshape shape');
        }
        
        // Calculate inputElementCount (product of all items in input's shape)
        const inputElementCount = inputShape.reduce((a, b) => a * b, 1);
        
        // Calculate known (product of all values in shape except -1)
        const known = shape.reduce((a, b) => b === -1 ? a : a * b, 1);
        
        if (known === 0) {
          throw new TypeError('Product of shape dimensions contains 0');
        }
        
        const idx = shape.indexOf(-1);
        const inferredDim = Math.floor(inputElementCount / known);
        
        // Check if the inferred dimension results in the same number of elements
        if (inferredDim * known !== inputElementCount) {
          throw new TypeError('Total size of input tensor is not divisible by product of specified dimensions');
        }
        
        shape[idx] = inferredDim;
      }
      
      // Validate the shape: ensure all values are valid unsigned long integers
      const outputShape = shape.map(dim => {
        if (isNaN(dim) || !isFinite(dim) || dim < 0) {
          throw new TypeError('Shape dimension must be a non-negative integer');
        }
        return Math.floor(Number(dim));
      });
      
      // Check if product of newShape equals inputElementCount
      const inputElementCount = inputShape.reduce((a, b) => a * b, 1);
      const outputElementCount = outputShape.reduce((a, b) => a * b, 1);
      
      if (outputElementCount !== inputElementCount) {
        throw new TypeError('Product of output shape dimensions must equal the product of input shape dimensions');
      }
      
      return outputShape;
    })()`;
  };

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

    // Generate code for shape processing
    const js_shape = generateShapeProcessingCode(array);
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
    const shapeArr = newShapeAttr.value;
    const js_shape = generateShapeProcessingCode(shapeArr.join(', '));
    const labelOpt = node.name ? `{ label: '${node.name}' }` : '{}';
    
    return `
    const ${outputVars[0]} = builder.reshape(
      ${inputVars[0]},
      ${js_shape},
      ${labelOpt}
    );`;
  }

  throw new Error('Reshape node missing shape input or new_shape attribute');
}