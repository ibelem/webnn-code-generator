/**
 * Generate JavaScript code for a WebNN softmax operation from ONNX Softmax node info.
 * https://www.w3.org/TR/webnn/#api-mlgraphbuilder-softmax-method
 */

import { getNonEmptyStringAroundNewline } from '../../utils';

export function softmax(
  node: any,
  toJsVarName: (name: string) => string
): string {
  const inputs: string[] = node.inputs?.map((i: any) => getNonEmptyStringAroundNewline(i.value?.[0]?.name)) || [];
  const outputs: string[] = node.outputs?.map((o: any) => getNonEmptyStringAroundNewline(o.value?.[0]?.name)) || [];
  const attrs: any[] = node.attributes || [];
  const inputVar = toJsVarName(inputs[0]);
  const outputVar = toJsVarName(outputs[0]);

  // Default axis is 1 for ONNX Softmax
  let axis = 1;
  for (const attr of attrs) {
    if (attr.name === 'axis') {
      // Handle ONNX JSON format: value: { type: "bigint", value: "-1" }
      if (attr.value && typeof attr.value.value === 'string') {
        axis = Number(attr.value.value);
      } else if (typeof attr.value === 'number') {
        axis = attr.value;
      }
      break;
    }
  }

  if (axis < 0) {
    // Get input shape from node.inputs[0].value[0].type.shape.dimensions or default to []
    const inputShape = node.inputs?.[0]?.value?.[0]?.type?.shape?.dimensions || [];
    axis = inputShape.length + axis;
  }

  return `
    const ${outputVar} = builder.softmax(
      ${inputVar},
      ${axis}
    );`;
}