import {
  getInputVars,
  getOutputVars,
  getAttr
} from '../../operation-utils';

/**
 * Generate JavaScript code for a WebNN split operation from ONNX Split node info.
 * https://www.w3.org/TR/webnn/#api-mlgraphbuilder-split
 * https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/webnn/builders/impl/split_op_builder.cc
 */
export function Split(
  node: any,
  toJsVarName: (name: string) => string
): string {
  const inputVars = getInputVars(node, toJsVarName); // [input, (optional) split]
  const outputVars = getOutputVars(node, toJsVarName);

  // Get axis (default 0, handle negative axis)
  const inputShape = node.inputs?.[0]?.shape;
  const rank = inputShape ? inputShape.length : undefined;
  let axis = getAttr(node, 'axis', 0);
  if (axis < 0 && rank !== undefined) axis += rank;

  // Get splits: can be a number (count) or an array (sizes)
  let splits = getAttr(node, 'split', undefined);
  let splitArg: string;
  if (Array.isArray(splits) && splits.length > 0) {
    splitArg = `[${splits.join(', ')}]`;
  } else if (typeof splits === 'number') {
    splitArg = splits.toString();
  } else if (node.inputs?.[1]?.initializer) {
    // If split is provided as an input initializer
    const splitValues = node.inputs[1].initializer.value;
    splitArg = `[${Array.from(splitValues).join(', ')}]`;
  } else {
    // Default: split into equal parts (number of outputs)
    splitArg = outputVars.length.toString();
  }

  // Add label for debugging if node.name exists
  const opts = node.name ? `{ axis: ${axis}, label: '${node.name}' }` : `{ axis: ${axis} }`;

  // Generate code for all outputs
  return `
    const ${outputVars.join(', ')} = builder.split(
      ${inputVars[0]},
      ${splitArg},
      ${opts}
    );
`;
}