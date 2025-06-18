import {
  getInputVars,
  getOutputVars,
  getShape,
  getAttrValue
} from '../../operation-utils';

/**
 * Generate JavaScript code for a WebNN gru operation from ONNX GRU node info.
 * https://www.w3.org/TR/webnn/#api-mlgraphbuilder-gru
 * https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/webnn/builders/impl/gru_op_builder.cc
 */
export function GRU(
  node: any,
  toJsVarName: (name: string) => string
): string {
  const inputVars = getInputVars(node, toJsVarName); // [input, weight, recurrentWeight, bias, sequence_lens, initialHiddenState]
  const outputVars = getOutputVars(node, toJsVarName);

  let direction = getAttrValue(node, 'direction', 'forward');
  let hiddenSize = getAttrValue(node, 'hidden_size', undefined);
  let linearBeforeReset = !!getAttrValue(node, 'linear_before_reset', 0);
  let activations = getAttrValue(node, 'activations', undefined);

  // Get steps from input shape
  const { shape: inputShape } = getShape(node, 0, false);
  const steps = inputShape[0];

  // Build options
  let options: string[] = [
    `label: '${node.name}'`,
    `layout: 'zrn'`
  ];

  // Bias and recurrentBias (split if present)
  if (inputVars[3]) {
    // ONNX bias shape: [numDirections, 6*hiddenSize], WebNN expects bias and recurrentBias separately
    // Split along axis 1 into [bias, recurrentBias]
    options.push(
      `...(() => {
        const splitted = builder.split(${inputVars[3]}, 2, {axis: 1, label: '${node.name}_split'});
        return {bias: splitted[0], recurrentBias: splitted[1]};
      })()`
    );
  }

  // initialHiddenState
  if (inputVars[5]) {
    options.push(`initialHiddenState: ${inputVars[5]}`);
  }

  // resetAfter (WebNN default true, ONNX linear_before_reset: 0=false, 1=true)
  options.push(`resetAfter: ${linearBeforeReset ? 'true' : 'false'}`);

  // returnSequence: Y (output 0) is the sequence, Y_h (output 1) is the last hidden state
  // ONNX: Y is optional, Y_h is required. WebNN: outputs[0]=Y_h, outputs[1]=Y if returnSequence.
  // We set returnSequence if Y is present.
  const hasY = !!outputVars[0];
  const hasYh = !!outputVars[1];
  if (hasY) options.push('returnSequence: true');
  else options.push('returnSequence: false');

  // direction
  let webnnDirection = 'forward';
  if (direction === 'forward') webnnDirection = 'forward';
  else if (direction === 'reverse') webnnDirection = 'backward';
  else if (direction === 'bidirectional') webnnDirection = 'both';
  options.push(`direction: '${webnnDirection}'`);

  // activations
  if (activations && activations.length >= 2) {
    // Map ONNX activations to WebNN
    const actMap: Record<string, string> = {Relu: 'relu', Sigmoid: 'sigmoid', Tanh: 'tanh'};
    const acts = activations.slice(0, 2).map((a: string) => `'${actMap[a] || a.toLowerCase()}'`);
    options.push(`activations: [${acts.join(', ')}]`);
  }

  // Compose options object
  const optionsObj = `{ ${options.join(',\n    ')} }`;

  // WebNN expects: gru(input, weight, recurrentWeight, steps, hiddenSize, options)
  // ONNX: inputVars = [input, weight, recurrentWeight, bias, sequence_lens, initialHiddenState]
  // sequence_lens is not supported by WebNN, skip inputVars[4]

  // Outputs: WebNN returns [Y_h, Y] if returnSequence, else [Y_h]
  // ONNX: output 0 = Y (optional), output 1 = Y_h (optional)
  // We assign accordingly
  let code = `
    const ${outputVars[0] || 'Y'}_gru_outputs = builder.gru(
      ${inputVars[0]}, // input
      ${inputVars[1]}, // weight
      ${inputVars[2]}, // recurrentWeight
      ${steps}, // steps
      ${hiddenSize}, // hiddenSize
      ${optionsObj}
    );
`;

  if (hasY && hasYh) {
    code += `
    const ${outputVars[0]} = ${outputVars[0]}_gru_outputs[1]; // Y (sequence)
    const ${outputVars[1]} = ${outputVars[0]}_gru_outputs[0]; // Y_h (last hidden)
`;
  } else if (hasY) {
    code += `
    const ${outputVars[0]} = ${outputVars[0]}_gru_outputs[1]; // Y (sequence)
`;
  } else if (hasYh) {
    code += `
    const ${outputVars[1]} = ${outputVars[0]}_gru_outputs[0]; // Y_h (last hidden)
`;
  }

  return code;
}