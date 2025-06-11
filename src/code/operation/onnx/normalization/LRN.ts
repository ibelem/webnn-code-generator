import {
  getInputVars,
  getOutputVars,
  getAttr
} from '../../operation-utils';

/**
 * Decompose ONNX LRN op into a series of WebNN ops.
 * https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/webnn/builders/impl/lrn_op_builder.cc
 * Implements: Y = X / (bias + (alpha / size) * sum(X^2))^beta
 */
export function LRN(
  node: any,
  toJsVarName: (name: string) => string
): string {
  const inputVars = getInputVars(node, toJsVarName); // [X]
  const outputVars = getOutputVars(node, toJsVarName); // [Y]

  // Get attributes with ONNX defaults
  const alpha = getAttr(node, 'alpha', 0.0001);
  const beta = getAttr(node, 'beta', 0.75);
  const bias = getAttr(node, 'bias', 1.0);
  const size = getAttr(node, 'size', 1);

  // Calculate paddings for channel axis (axis=1, NCHW)
  const leadingPadding = Math.floor((size - 1) / 2);
  const trailingPadding = Math.ceil((size - 1) / 2);

  // Compose code for decomposition
  return `
    // == ${node.name} · start ==
    // Decompose ${node.name} into WebNN ops
    const ${outputVars[0]}_pow1 = builder.pow(
      ${inputVars[0]},
      builder.constant(
        {type: 'float32', shape: []}, new Float32Array([2])
      ),
      {label: '${node.name}_pow1'}
    );
    const ${outputVars[0]}_pad = builder.pad(
      ${outputVars[0]}_pow1,
      [0, 0, 0, ${leadingPadding}],
      [0, 0, 0, ${trailingPadding}],
      {label: '${node.name}_pad'}
    );
    const ${outputVars[0]}_pool = builder.averagePool2d(
      ${outputVars[0]}_pad,
      {windowDimensions: [1, ${size}], label: '${node.name}_avgpool'}
    );
    const ${outputVars[0]}_mul = builder.mul(
      ${outputVars[0]}_pool,
      builder.constant({type: 'float32', shape: []}, new Float32Array([${alpha}])),
      {label: '${node.name}_mul'}
    );
    const ${outputVars[0]}_add = builder.add(
      ${outputVars[0]}_mul,
      builder.constant({type: 'float32', shape: []}, new Float32Array([${bias}])),
      {label: '${node.name}_add'}
    );
    const ${outputVars[0]}_pow2 = builder.pow(
      ${outputVars[0]}_add,
      builder.constant({type: 'float32', shape: []}, new Float32Array([${beta}])),
      {label: '${node.name}_pow2'}
    );
    const ${outputVars[0]} = builder.div(
      ${inputVars[0]},
      ${outputVars[0]}_pow2,
      {label: '${node.name}_div'}
    );
    // == ${node.name} · end ==
`;
}