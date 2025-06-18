import {
  getInputVars,
  getOutputVars,
  getAttrValue,
  getShape
} from '../../operation-utils';

/**
 * Generate JavaScript code for a WebNN cumulativeSum operation from ONNX CumSum node info.
 */
export function CumSum(
  node: any,
  toJsVarName: (name: string) => string
): string {
  const inputVars = getInputVars(node, toJsVarName);
  const outputVars = getOutputVars(node, toJsVarName);

  // Use getAttrValue for robust attribute extraction
  let axis = getAttrValue(node, 'axis', 0);
  const exclusive = !!getAttrValue(node, 'exclusive', 0);
  const reverse = !!getAttrValue(node, 'reverse', 0);

  // Try to get input rank for negative axis handling
  let inputRank = 0;
  if (node.inputs && node.inputs.length > 0) {
    const { shape } = getShape(node, 0, false);
    if (Array.isArray(shape)) inputRank = shape.length;
  }

  // Handle negative axis
  if (axis < 0 && inputRank > 0) {
    axis = inputRank + axis;
  }

  const opts = [
    `axis: ${axis}`,
    exclusive ? `exclusive: true` : '',
    reverse ? `reversed: true` : ''
  ].filter(Boolean);
  if (node.name) opts.push(`label: '${node.name}'`);

  return `
    const ${outputVars[0]} = builder.cumulativeSum(
      ${inputVars[0]},
      { ${opts.join(', ')} }
    );`;
}