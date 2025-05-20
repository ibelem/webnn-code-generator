/**
 * Generate JavaScript code for a WebNN conv2d operation from ONNX node info.
 * @param node - The ONNX node object (with inputs, outputs, attributes)
 * @param toJsVarName - Function to convert ONNX names to JS variable names
 * @returns JavaScript code string for the conv2d operation
 */
export function conv2d_js(
  node: any,
  toJsVarName: (name: string) => string
): string {
  const inputs: string[] = node.inputs?.map((i: any) => i.value?.[0].name) || [];
  const outputs: string[] = node.outputs?.map((o: any) => o.value?.[0].name) || [];
  const attrs: any[] = node.attributes || [];

  // Map attribute array to a dictionary by name
  const attrDict: Record<string, any> = {};
  for (const attr of attrs) {
    attrDict[attr.name] = attr;
  }

  const inputVars = inputs.map(toJsVarName);
  const outputVar = toJsVarName(outputs[0]);

  // Handle strides
  let strides_js = 'undefined';
  const strides = attrDict['strides']?.value?.value;
  if (Array.isArray(strides) && strides.length === 2) {
    strides_js = `[${strides.map((s: any) => String(Number(s))).join(', ')}]`;
  }

  // Handle pads (convert ONNX [x1_begin, x2_begin, x1_end, x2_end] to WebNN [top, left, bottom, right])
  let pads_js = 'undefined';
  const pads = attrDict['pads']?.value?.value;
  if (Array.isArray(pads) && pads.length === 4) {
    // ONNX: [top, left, bottom, right] is [0,1,2,3] in ONNX
    // WebNN expects [top, bottom, left, right]
    // But in your python you use [0,2,1,3] (top, bottom, left, right)
    const pads_webnn = [pads[0], pads[2], pads[1], pads[3]];
    pads_js = `[${pads_webnn.map((p: any) => String(Number(p))).join(', ')}]`;
  }

  // Handle dilations
  let dilations_js = 'undefined';
  const dilations = attrDict['dilations']?.value?.value;
  if (Array.isArray(dilations) && dilations.length === 2) {
    dilations_js = `[${dilations.map((d: any) => String(Number(d))).join(', ')}]`;
  }

  // Handle groups
  const groups = attrDict['group']?.value?.value;
  const groups_js = groups !== undefined ? String(Number(groups)) : 'undefined';

  // Bias input (optional)
  const biasVar = inputVars.length > 2 ? inputVars[2] : 'undefined';

  return `
    const ${outputVar} = builder.conv2d(
      ${inputVars[0]}, ${inputVars[1]},
      {
        bias: ${biasVar},
        strides: ${strides_js},
        padding: ${pads_js},
        dilations: ${dilations_js},
        groups: ${groups_js}
      }
    );`;
}