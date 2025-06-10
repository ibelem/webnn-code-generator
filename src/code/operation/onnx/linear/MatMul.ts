import {
  getInputVars,
  getOutputVars
} from '../../operation-utils';

/**
 * Generate JavaScript code for WebNN matmul operation from ONNX node info.
 * Handles 1D input promotion as required by ONNX spec
 * https://www.w3.org/TR/webnn/#api-mlgraphbuilder-matmul
 * https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/webnn/builders/impl/gemm_op_builder.cc
 */
export function MatMul(
  node: any,
  toJsVarName: (name: string) => string
): string {
  const inputVars = getInputVars(node, toJsVarName);
  const outputVars = getOutputVars(node, toJsVarName);

  // Try to get input shapes if available
  const aShape = node.inputs?.[0]?.value?.[0]?.type?.shape?.dimensions;
  const bShape = node.inputs?.[1]?.value?.[0]?.type?.shape?.dimensions;
  const outputShape = node.outputs?.[0]?.value?.[0]?.type?.shape?.dimensions;

  let aVar = inputVars[0];
  let bVar = inputVars[1];
  let code = '';

  // Promote 1D inputs to 2D as per ONNX spec
  if (Array.isArray(aShape) && aShape.length === 1) {
    const reshapedA = `${aVar}_reshaped`;
    code += `
    // Promote 1D A to 2D
    const ${reshapedA} = builder.reshape(${aVar}, [1, ${aShape[0]}], { label: '${node.name ?? ''}_reshape_a' });`;
    aVar = reshapedA;
  }
  if (Array.isArray(bShape) && bShape.length === 1) {
    const reshapedB = `${bVar}_reshaped`;
    code += `
    // Promote 1D B to 2D
    const ${reshapedB} = builder.reshape(${bVar}, [${bShape[0]}, 1], { label: '${node.name ?? ''}_reshape_b' });`;
    bVar = reshapedB;
  }

  // MatMul with label
  code += `
    let ${outputVars[0]} = builder.matmul(
      ${aVar},
      ${bVar},
      { label: '${node.name ?? ''}' }
    );`;

  // If either input was 1D, reshape output back to ONNX spec
  if (
    (Array.isArray(aShape) && aShape.length === 1) ||
    (Array.isArray(bShape) && bShape.length === 1)
  ) {
    if (Array.isArray(outputShape)) {
      code += `
    // Reshape output back to original ONNX output shape
    ${outputVars[0]} = builder.reshape(${outputVars[0]}, [${outputShape.join(', ')}], { label: '${node.name ?? ''}_reshape_output' });`;
    }
  }

  return code;
}