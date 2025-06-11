import {
  getInputVars,
  getOutputVars,
  getAttr
} from '../../operation-utils';

/**
 * Decompose ONNX LRN op into a series of WebNN ops.
 * https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/webnn/builders/impl/lrn_op_builder.cc
 * Implements: Y = X / (bias + (alpha / size) * sum(X^2))^beta
 * 
 * Note: LRN is layout sensitive - normalization happens across the channel dimension.
 */
export function LRN(
  node: any,
  toJsVarName: (name: string) => string,
  options: { nhwc?: boolean } = {}
): string {
  const inputVars = getInputVars(node, toJsVarName); // [X]
  const outputVars = getOutputVars(node, toJsVarName); // [Y]
  const nhwc = !!options.nhwc;
  
  // Get attributes with ONNX defaults
  const alpha = getAttr(node, 'alpha', 0.0001);
  const beta = getAttr(node, 'beta', 0.75);
  const bias = getAttr(node, 'bias', 1.0);
  const size = getAttr(node, 'size', 1);

  // In NCHW, channels are at axis=1
  // In NHWC, channels are at axis=3
  const channelAxis = nhwc ? 3 : 1;
  
  // Calculate paddings for channel axis
  const leadingPadding = Math.floor((size - 1) / 2);
  const trailingPadding = Math.ceil((size - 1) / 2);
  
  // Create padding arrays based on layout
  const beginPadding = [0, 0, 0, 0].map((_, i) => i === channelAxis ? leadingPadding : 0);
  const endPadding = [0, 0, 0, 0].map((_, i) => i === channelAxis ? trailingPadding : 0);
  
  // Window dimensions are 1s except for channel dimension
  const windowDimensions = [1, 1, 1, 1];
  windowDimensions[channelAxis] = size;

  // layout-specific pooling options
  const layoutOption = nhwc ? 'nhwc' : 'nchw';
  
  // Compose code for decomposition
  return `
    // == ${node.name} · start ==
    // Decompose ${node.name} into WebNN ops (${nhwc ? 'NHWC' : 'NCHW'} layout)
    const ${outputVars[0]}_pow1 = builder.pow(
      ${inputVars[0]},
      builder.constant(
        {type: 'float32', shape: []}, new Float32Array([2])
      ),
      {label: '${node.name}_pow1'}
    );
    const ${outputVars[0]}_pad = builder.pad(
      ${outputVars[0]}_pow1,
      [${beginPadding.join(', ')}],
      [${endPadding.join(', ')}],
      {label: '${node.name}_pad'}
    );
    const ${outputVars[0]}_pool = builder.averagePool2d(
      ${outputVars[0]}_pad,
      {
        windowDimensions: [${windowDimensions[1]}, ${windowDimensions[2]}],
        layout: '${layoutOption}',
        label: '${node.name}_avgpool'
      }
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