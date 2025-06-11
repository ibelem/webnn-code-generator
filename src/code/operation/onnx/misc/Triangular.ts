import {
  getInputVars,
  getOutputVars
} from '../../operation-utils';

/**
 * Generate JavaScript code for WebNN triangular operation from ONNX Triangular node.
 * https://www.w3.org/TR/webnn/#api-mlgraphbuilder-triangular
 * https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/webnn/builders/impl/triangular_op_builder.cc
 */
export function Triangular(
  node: any,
  toJsVarName: (name: string) => string
): string {
  const inputVars = getInputVars(node, toJsVarName); // [input, diagonal?]
  const outputVars = getOutputVars(node, toJsVarName);

  // Default options
  let upper = true;
  let diagonal: number | undefined = undefined;

  // Parse attributes for upper and diagonal
  if (node.attributes) {
    for (const attr of node.attributes) {
      if (attr.name === 'upper') {
        upper = !!(typeof attr.value === 'boolean' ? attr.value : Number(attr.value));
      }
      if (attr.name === 'diagonal') {
        diagonal = typeof attr.value === 'number'
          ? attr.value
          : (Array.isArray(attr.value) ? attr.value[0] : attr.value?.value ?? undefined);
      }
    }
  }

  // If diagonal is provided as input[1] (initializer)
  if (node.inputs.length > 1 && node.inputs[1]?.value?.[0]?.initializer) {
    const init = node.inputs[1].value[0].initializer;
    const keys = Object.keys(init.values).sort((a, b) => Number(a) - Number(b));
    if (keys.length > 0) {
      diagonal = Number(init.values[keys[0]]);
    }
  }

  // Compose options
  const opts: string[] = [`upper: ${upper}`];
  if (typeof diagonal === 'number') {
    opts.push(`diagonal: ${diagonal}`);
  }
  if (node.name) {
    opts.push(`label: '${node.name}'`);
  }

  return `
    const ${outputVars[0]} = builder.triangular(
      ${inputVars[0]},
      { ${opts.join(', ')} }
    );
`;
}