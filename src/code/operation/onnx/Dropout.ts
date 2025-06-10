import {
  getInputVars,
  getOutputVars
} from '../operation-utils';

/**
 * Generate JavaScript code for WebNN from ONNX Dropout node.
 * https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/webnn/builders/impl/dropout_op_builder.cc
 * Only supports inference (test) mode: just identity.
 * If mask output is requested, returns a constant tensor of all ones (bool).
 */
export function Dropout(
  node: any,
  toJsVarName: (name: string) => string
): string {
  const inputVars = getInputVars(node, toJsVarName);
  const outputVars = getOutputVars(node, toJsVarName);

  let code = `
      const ${outputVars[0]} = builder.identity(
        ${inputVars[0]},
        { label: '${node.name || ''}' }
      );`;

  // If mask output is requested (Dropout has 2 outputs)
  if (outputVars.length > 1 && node.outputs?.[1]?.shape) {
    const maskShape = node.outputs[1].shape;
    code += `
      const ${outputVars[1]}_ones = builder.constant(
        {type: 'bool', shape: [${maskShape.join(', ')}]},
        new Uint8Array(${maskShape.reduce((a: number, b: number) => a * b, 1)}).fill(1)
      );
      const ${outputVars[1]} = builder.identity(
        ${outputVars[1]}_ones,
        { label: '${outputVars[1]}_identity' }
      );`;
  }

  return code;
}