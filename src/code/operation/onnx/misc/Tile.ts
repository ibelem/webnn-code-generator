import {
  getInputVars,
  getOutputVars
} from '../../operation-utils';

/**
 * Generate JavaScript code for WebNN tile operation from ONNX Tile node.
 * https://www.w3.org/TR/webnn/#api-mlgraphbuilder-tile
 * https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/webnn/builders/impl/tile_op_builder.cc
 */
export function Tile(
  node: any,
  toJsVarName: (name: string) => string
): string {
  const inputVars = getInputVars(node, toJsVarName); // [input, repeats]
  const outputVars = getOutputVars(node, toJsVarName);

  // Get repetitions from input[1] initializer
  let repetitions: number[] = [];
  if (node.inputs.length > 1 && node.inputs[1]?.value?.[0]?.initializer) {
    const init = node.inputs[1].value[0].initializer;
    repetitions = Object.keys(init.values)
      .sort((a, b) => Number(a) - Number(b))
      .map(k => Number(init.values[k]));
  }

  const labelOpt = node.name ? `{ label: '${node.name}' }` : '{}';

  return `
    const ${outputVars[0]} = builder.tile(
      ${inputVars[0]},
      [${repetitions.join(', ')}],
      ${labelOpt}
    );
`;
}