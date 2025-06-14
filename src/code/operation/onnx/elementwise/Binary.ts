import {
  getInputVars,
  getOutputVars
} from '../../operation-utils';

/**
 * Generate JavaScript code for a WebNN binary operation (add, sub, mul, div, max, min, pow, prelu).
 * https://www.w3.org/TR/webnn/#api-mlgraphbuilder-binary
 * https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/webnn/builders/impl/binary_op_builder.cc
 * https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/webnn/builders/impl/max_min_op_builder.cc
 */
function Binary(
  node: any,
  toJsVarName: (name: string) => string,
  options: { [key: string]: any } = {}
): string {
  const inputVars = getInputVars(node, toJsVarName);
  const outputVars = getOutputVars(node, toJsVarName);
  const opType = options.opType;

  // Add label for debugging
  const opts = node.name ? `{ label: '${node.name}' }` : '';

  return `
    const ${outputVars[0]} = builder.${opType}(
      ${inputVars[0]},
      ${inputVars[1]},
      ${opts}
    );`;
}

export function Add(node: any, toJsVarName: (name: string) => string, options: { [key: string]: any } = {}): string {
  return Binary(node, toJsVarName, { ...options, opType: 'add' });
}
export function Sub(node: any, toJsVarName: (name: string) => string, options: { [key: string]: any } = {}): string {
  return Binary(node, toJsVarName, { ...options, opType: 'sub' });
}
export function Mul(node: any, toJsVarName: (name: string) => string, options: { [key: string]: any } = {}): string {
  return Binary(node, toJsVarName, { ...options, opType: 'mul' });
}
export function Div(node: any, toJsVarName: (name: string) => string, options: { [key: string]: any } = {}): string {
  return Binary(node, toJsVarName, { ...options, opType: 'div' });
}

function MultiInputMinMax(
  node: any,
  toJsVarName: (name: string) => string,
  opType: 'min' | 'max'
): string {
  const inputVars = getInputVars(node, toJsVarName);
  const outputVars = getOutputVars(node, toJsVarName);
  const label = node.name ? `{ label: '${node.name}' }` : '';

  if (inputVars.length === 1) {
    return `
    const ${outputVars[0]} = builder.identity(
      ${inputVars[0]},
      ${label}
    );`;
  }

  let code = `
    let ${outputVars[0]} = builder.${opType}(
      ${inputVars[0]},
      ${inputVars[1]},
      ${label}
    );`;

  for (let i = 2; i < inputVars.length; ++i) {
    const nextLabel = node.name ? `{ label: '${node.name}_input${i}' }` : '';
    code += `
    ${outputVars[0]} = builder.${opType}(
      ${outputVars[0]},
      ${inputVars[i]},
      ${nextLabel}
    );`;
  }
  return code;
}

export function Max(node: any, toJsVarName: (name: string) => string): string {
  return MultiInputMinMax(node, toJsVarName, 'max');
}
export function Min(node: any, toJsVarName: (name: string) => string): string {
  return MultiInputMinMax(node, toJsVarName, 'min');
}
export function Pow(node: any, toJsVarName: (name: string) => string, options: { [key: string]: any } = {}): string {
  return Binary(node, toJsVarName, { ...options, opType: 'pow' });
}
