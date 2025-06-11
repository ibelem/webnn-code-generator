import {
  getInputVars,
  getOutputVars
} from '../../operation-utils';

/**
 * Generate JavaScript code for a WebNN lstm operation from ONNX LSTM node info.
 * https://www.w3.org/TR/webnn/#api-mlgraphbuilder-lstm
 * https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/webnn/builders/impl/lstm_op_builder.cc
 */
export function LSTM(
  node: any,
  toJsVarName: (name: string) => string
): string {
  const inputVars = getInputVars(node, toJsVarName); // [input, weight, recurrentWeight, bias, sequence_lens, initialHiddenState, initialCellState, peepholeWeight]
  const outputVars = getOutputVars(node, toJsVarName);

  // Required attributes
  let hiddenSize = 0;
  let direction = 'forward';
  let activations: string[] | undefined = undefined;

  for (const attr of node.attributes || []) {
    if (attr.name === 'hidden_size') hiddenSize = Number(attr.value);
    if (attr.name === 'direction') direction = attr.value;
    if (attr.name === 'activations') activations = attr.value;
  }

  // Get steps from input shape
  const inputShape = node.inputs?.[0]?.shape || [];
  const steps = inputShape[0];

  // Build options
  let options: string[] = [
    `label: '${node.name}'`,
    `layout: 'iofg'`
  ];

  // Bias and recurrentBias (split if present)
  if (inputVars[3]) {
    // ONNX bias shape: [numDirections, 8*hiddenSize], WebNN expects bias and recurrentBias separately
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

  // initialCellState
  if (inputVars[6]) {
    options.push(`initialCellState: ${inputVars[6]}`);
  }

  // peepholeWeight
  if (inputVars[7]) {
    options.push(`peepholeWeight: ${inputVars[7]}`);
  }

  // direction
  let webnnDirection = 'forward';
  if (direction === 'forward') webnnDirection = 'forward';
  else if (direction === 'reverse') webnnDirection = 'backward';
  else if (direction === 'bidirectional') webnnDirection = 'both';
  options.push(`direction: '${webnnDirection}'`);

  // Outputs: ONNX Y (sequence), Y_h (last hidden), Y_c (last cell)
  // WebNN: [Y_h, Y_c, Y] if returnSequence, else [Y_h, Y_c]
  const hasY = !!outputVars[0];
  const hasYh = !!outputVars[1];
  const hasYc = !!outputVars[2];
  options.push(`returnSequence: ${hasY ? 'true' : 'false'}`);

  // activations
  if (activations && activations.length >= 3) {
    // Map ONNX activations to WebNN
    const actMap: Record<string, string> = {Relu: 'relu', Sigmoid: 'sigmoid', Tanh: 'tanh'};
    const acts = activations.slice(0, 3).map((a: string) => `'${actMap[a] || a.toLowerCase()}'`);
    options.push(`activations: [${acts.join(', ')}]`);
  }

  // Compose options object
  const optionsObj = `{ ${options.join(',\n    ')} }`;

  // WebNN expects: lstm(input, weight, recurrentWeight, steps, hiddenSize, options)
  // ONNX: inputVars = [input, weight, recurrentWeight, bias, sequence_lens, initialHiddenState, initialCellState, peepholeWeight]
  // sequence_lens is not supported by WebNN, skip inputVars[4]

  // Outputs: WebNN returns [Y_h, Y_c, Y] if returnSequence, else [Y_h, Y_c]
  // ONNX: output 0 = Y (optional), output 1 = Y_h (optional), output 2 = Y_c (optional)
  // We assign accordingly
  let code = `
    const ${outputVars[0] || 'Y'}_lstm_outputs = builder.lstm(
      ${inputVars[0]}, // input
      ${inputVars[1]}, // weight
      ${inputVars[2]}, // recurrentWeight
      ${steps}, // steps
      ${hiddenSize}, // hiddenSize
      ${optionsObj}
    );
`;

  if (hasY && hasYh && hasYc) {
    code += `
    const ${outputVars[0]} = ${outputVars[0]}_lstm_outputs[2]; // Y (sequence)
    const ${outputVars[1]} = ${outputVars[0]}_lstm_outputs[0]; // Y_h (last hidden)
    const ${outputVars[2]} = ${outputVars[0]}_lstm_outputs[1]; // Y_c (last cell)
`;
  } else if (hasY && hasYh) {
    code += `
    const ${outputVars[0]} = ${outputVars[0]}_lstm_outputs[2]; // Y (sequence)
    const ${outputVars[1]} = ${outputVars[0]}_lstm_outputs[0]; // Y_h (last hidden)
`;
  } else if (hasY && hasYc) {
    code += `
    const ${outputVars[0]} = ${outputVars[0]}_lstm_outputs[2]; // Y (sequence)
    const ${outputVars[2]} = ${outputVars[0]}_lstm_outputs[1]; // Y_c (last cell)
`;
  } else if (hasYh && hasYc) {
    code += `
    const ${outputVars[1]} = ${outputVars[0]}_lstm_outputs[0]; // Y_h (last hidden)
    const ${outputVars[2]} = ${outputVars[0]}_lstm_outputs[1]; // Y_c (last cell)
`;
  } else if (hasY) {
    code += `
    const ${outputVars[0]} = ${outputVars[0]}_lstm_outputs[2]; // Y (sequence)
`;
  } else if (hasYh) {
    code += `
    const ${outputVars[1]} = ${outputVars[0]}_lstm_outputs[0]; // Y_h (last hidden)
`;
  } else if (hasYc) {
    code += `
    const ${outputVars[2]} = ${outputVars[0]}_lstm_outputs[1]; // Y_c (last cell)
`;
  }

  return code;
}