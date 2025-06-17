import {
  getInputVars,
  getOutputVars,
  getAttrValue
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
  const inputVars = getInputVars(node, toJsVarName);
  const outputVars = getOutputVars(node, toJsVarName);

  // Use getAttrValue for robust attribute extraction
  const upper = !!getAttrValue(node, 'upper', 1); // default true
  const unitriangular = !!getAttrValue(node, 'unitriangular', 0); // default false
  const diagonal = getAttrValue(node, 'diagonal', 0); // default 0

  const opts: string[] = [];
  if (!upper) opts.push(`upper: false`);
  if (unitriangular) opts.push(`unitriangular: true`);
  if (diagonal !== 0) opts.push(`diagonal: ${diagonal}`);
  if (node.name) opts.push(`label: '${node.name}'`);

  const optsString = opts.length ? `{ ${opts.join(', ')} }` : '';

  return `
    const ${outputVars[0]} = builder.triangular(
      ${inputVars[0]},
      ${optsString}
    );
  `;
}