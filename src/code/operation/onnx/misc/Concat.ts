import {
  getInputVars,
  getOutputVars,
  getAttrValue
} from '../../operation-utils';

/**
 * Generate JavaScript code for a WebNN concat operation from ONNX Concat node info.
 * https://www.w3.org/TR/webnn/#api-mlgraphbuilder-concat
 * https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/webnn/builders/impl/concat_op_builder.cc
 */
export function Concat(
  node: any,
  toJsVarName: (name: string) => string
): string {
  const inputVars = getInputVars(node, toJsVarName);
  const outputVars = getOutputVars(node, toJsVarName);

  // Use getAttrValue for robust attribute extraction
  let axis = getAttrValue(node, 'axis', 0);

  // Generate code that handles negative axis and ensures unsigned long at runtime
  const opts = [];
  if (node.name) opts.push(`label: '${node.name}'`);

  return `
    // Handle negative axis and ensure unsigned long for WebNN API
    let axis_${outputVars[0]} = ${axis};
    // If axis is negative, convert to positive based on input rank
    if (axis_${outputVars[0]} < 0) {
      // Use the first input's rank to resolve negative axis
      const firstInputRank = ${inputVars[0]}.shape.length;
      axis_${outputVars[0]} = firstInputRank + axis_${outputVars[0]};
    }
    // Ensure axis is a non-negative integer (unsigned long) as required by WebNN API
    axis_${outputVars[0]} = Math.max(0, Math.floor(axis_${outputVars[0]}));

    const ${outputVars[0]} = builder.concat(
      [${inputVars.join(', ')}],
      axis_${outputVars[0]},
      { ${opts.join(', ')} }
    );`;
}