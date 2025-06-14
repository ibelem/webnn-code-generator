import {
  getInputVars,
  getOutputVars,
  getShape
} from '../../operation-utils';

/**
 * Generate JavaScript code for a WebNN slice operation from ONNX Slice node info.
 * Handles negative steps (reverse), axes, and default values.
 * https://www.w3.org/TR/webnn/#api-mlgraphbuilder-slice
 * https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/webnn/builders/impl/slice_op_builder.cc
 */
export function Slice(
  node: any,
  toJsVarName: (name: string) => string,
  options: { [key: string]: any } = {}
): string {
  const nhwc = !!options.nhwc;
  const inputVars = getInputVars(node, toJsVarName);
  const outputVars = getOutputVars(node, toJsVarName);
  const inputShape = getShape(node, 0, nhwc);

  // Helper to extract initializer array from input index
  function getInitializerArr(idx: number): number[] | undefined {
    if (node.inputs.length > idx && node.inputs[idx]?.value?.[0]?.initializer) {
      const init = node.inputs[idx].value[0].initializer;
      const arr = Object.keys(init.values)
        .sort((a, b) => Number(a) - Number(b))
        .map(k => init.values[k]);
      return arr;
    }
    return undefined;
  }

  const rank = inputShape.length;
  let starts = getInitializerArr(1) || [];
  let ends = getInitializerArr(2) || [];
  let axes = getInitializerArr(3);
  let steps = getInitializerArr(4);

  // Default axes: [0, 1, ..., starts.length-1]
  if (!axes) axes = Array.from({ length: starts.length }, (_, i) => i);
  // Default steps: all 1
  if (!steps) steps = Array(starts.length).fill(1);

  // Prepare full-rank arrays for starts, ends, steps
  const fullStarts = Array(rank).fill(0);
  const fullEnds = inputShape.slice();
  const fullSteps = Array(rank).fill(1);

  for (let i = 0; i < axes.length; ++i) {
    const axis = axes[i];
    // Handle negative indices for starts/ends
    fullStarts[axis] = starts[i] < 0 ? inputShape[axis] + starts[i] : starts[i];
    fullEnds[axis] = ends[i] < 0 ? inputShape[axis] + ends[i] : ends[i];
    fullSteps[axis] = steps[i];
  }

  // Handle negative steps (reverse)
  const reverseAxes: number[] = [];
  for (let i = 0; i < rank; ++i) {
    if (fullSteps[i] < 0) {
      reverseAxes.push(i);
      fullSteps[i] = -fullSteps[i];
      fullStarts[i] = inputShape[i] - 1 - fullStarts[i];
      fullEnds[i] = inputShape[i] - 1 - fullEnds[i];
    }
  }

  // Compute sizes for WebNN slice, ensure non-negative
  const sizes = fullEnds.map((end, i) => Math.max(0, end - fullStarts[i]));

  // Only emit slice if needed
  const isSliceRequired = fullSteps.some((s, i) => s !== 1 || fullStarts[i] !== 0 || fullEnds[i] !== inputShape[i]);

  let code = '';
  let inputExpr = inputVars[0];

  if (reverseAxes.length > 0) {
    code += `
    const ${outputVars[0]}_reversed = builder.reverse(
      ${inputVars[0]},
      { axes: [${reverseAxes.join(', ')}], label: '${node.name || ''}_reverse' }
    );
`;
    inputExpr = `${outputVars[0]}_reversed`;
  }

  if (isSliceRequired) {
    code += `
    const ${outputVars[0]} = builder.slice(
      ${inputExpr},
      [${fullStarts.join(', ')}],
      [${sizes.join(', ')}],
      { strides: [${fullSteps.join(', ')}], label: '${node.name || ''}' }
    );
`;
  } else {
    code += `
    const ${outputVars[0]} = ${inputExpr};
`;
  }

  return code;
}