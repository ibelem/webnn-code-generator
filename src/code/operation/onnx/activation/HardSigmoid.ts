import {
  getInputVars,
  getOutputVars
} from '../../operation-utils';

/**
 * Generate JavaScript code for a WebNN hardSigmoid operation from ONNX HardSigmoid node info.
 * https://www.w3.org/TR/webnn/#api-mlgraphbuilder-hard-sigmoid
 */
export function HardSigmoid(
  node: any,
  toJsVarName: (name: string) => string
): string {
  const inputVars = getInputVars(node, toJsVarName);
  const outputVars = getOutputVars(node, toJsVarName);

  // Default values per ONNX spec: alpha=0.2, beta=0.5
  let alpha = 0.2;
  let beta = 0.5;
  for (const attr of node.attributes || []) {
    // Todo: Handle attr.f attribute correctly from json file
    if (attr.name === 'alpha') {
      alpha =
        typeof attr.f === 'number'
          ? attr.f
          : (typeof attr.value === 'number'
              ? attr.value
              : Number(attr.value?.value ?? 0.2));
    }
    if (attr.name === 'beta') {
      beta =
        typeof attr.f === 'number'
          ? attr.f
          : (typeof attr.value === 'number'
              ? attr.value
              : Number(attr.value?.value ?? 0.5));
    }
  }

  // Add label for debugging if node.name exists
  const opts = `{ alpha: ${alpha}, beta: ${beta}, label: '${node.name || ''}' }`;

  return `
    const ${outputVars[0]} = builder.hardSigmoid(
      ${inputVars[0]},
      ${opts}
    );
  `;
}