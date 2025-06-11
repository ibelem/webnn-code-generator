import {
  getInputVars,
  getOutputVars
} from '../../operation-utils';

/**
 * Generate JavaScript code for a WebNN elu operation from ONNX Elu node info.
 * https://www.w3.org/TR/webnn/#api-mlgraphbuilder-elu
 * https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/webnn/builders/impl/activation_op_builder.cc
 */

export function Elu(
  node: any,
  toJsVarName: (name: string) => string
): string {
  const inputVars = getInputVars(node, toJsVarName);
  const outputVars = getOutputVars(node, toJsVarName);

  // Default alpha is 1.0 for ONNX Elu
  let alpha = 1.0;
  for (const attr of node.attributes || []) {
    // Todo: Handle attr.f attribute correctly from json file
    if (attr.name === 'alpha') {
      alpha =
        typeof attr.f === 'number'
          ? attr.f
          : (typeof attr.value === 'number'
              ? attr.value
              : Number(attr.value?.value ?? 1.0));
      break;
    }
  }

  // Add label for debugging if node.name exists
  const opts = `{ alpha: ${alpha}, label: '${node.name || ''}' }`;

  return `
    const ${outputVars[0]} = builder.elu(
      ${inputVars[0]},
      ${opts}
    );
  `;
}