import {
  getInputVars,
  getOutputVars
} from '../operation-utils';

/**
 * Generate JavaScript code for a WebNN concat operation from ONNX Concat node info.
 * https://www.w3.org/TR/webnn/#api-mlgraphbuilder-concat
 */
export function Concat(
  node: any,
  toJsVarName: (name: string) => string
): string {
  const inputVars = getInputVars(node, toJsVarName);
  const outputVars = getOutputVars(node, toJsVarName);

  // ONNX axis is usually in node.attributes[0].value or node.attributes.find(a => a.name === 'axis')
  let axis = 0;
  for (const attr of node.attributes || []) {
    if (attr.name === 'axis') {
      if (typeof attr.value === 'number') {
        axis = attr.value;
      } else if (typeof attr.value?.value === 'number') {
        axis = attr.value.value;
      }
      break;
    }
  }

  return `
    const ${outputVars[0]} = builder.concat(
      [${inputVars.join(', ')}],
      ${axis}
    );`;
}