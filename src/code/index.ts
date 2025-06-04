// import { getModelState } from './ui';
import {
  modelName, getTypedArrayName, toJsVarName, hasKeysandNumberValues,
  getNonEmptyStringAroundNewline, findWeightNodeByName, downloadFile
} from '../utils';
import { getModelState, freeDimsOverrides } from '../ui';
import { opHandlers } from './operation';

function constructorCode() {
  return `
  constructor() {
    this.context_ = null;
    this.graph_ = null;
    this.inputTensors_ = {};
    this.outputTensors_ = {};
  }`;
}

function runCode() {
  return `
  async run(inputs) {
    // Set input buffers to input tensors using writeTensor (sync)
    for (const name in inputs) {
      if (!(name in this.inputTensors_)) throw new Error('Unknown input: ' + name);
      this.context_.writeTensor(this.inputTensors_[name], inputs[name]);
    }

    // Compute the graph
    await this.context_.dispatch(this.graph_, this.inputTensors_, this.outputTensors_);

    // Read output tensors to buffers using readTensor (async)
    const outputs = {};
    for (const name in this.outputTensors_) {
      const tensor = this.outputTensors_[name];
      const buffer = await this.context_.readTensor(tensor);
      let typedArrayCtor;
      switch (tensor.dataType) {
        case 'float32': typedArrayCtor = Float32Array; break;
        case 'uint8': typedArrayCtor = Uint8Array; break;
        case 'int8': typedArrayCtor = Int8Array; break;
        case 'uint16': typedArrayCtor = Uint16Array; break;
        case 'int16': typedArrayCtor = Int16Array; break;
        case 'int32': typedArrayCtor = Int32Array; break;
        case 'int64': typedArrayCtor = BigInt64Array; break;
        case 'float16': typedArrayCtor = Float16Array; break;
        case 'float64': typedArrayCtor = Float64Array; break;
        case 'uint32': typedArrayCtor = Uint32Array; break;
        case 'uint64': typedArrayCtor = BigUint64Array; break;
        default: throw new Error('Unhandled tensor dataType: ' + tensor.dataType);
      }
      outputs[name] = new typedArrayCtor(buffer);
    }
    return outputs;
  }`;
}

function buildCodeWithLayout(nhwc: boolean) {
  const { graphModelData, weightNchwModelData, weightNhwcModelData } = getModelState();
  const weightModelData = nhwc ? weightNhwcModelData : weightNchwModelData;
  const inputs = graphModelData?.graph?.[0].inputs;
  const outputs = graphModelData?.graph?.[0].outputs;

  let inputsCode = ``;
  for (const input of inputs) {
    const name = getNonEmptyStringAroundNewline(input.value[0]?.name);
    const dataType = input.value[0]?.type?.dataType;
    const shapeArr = input.value[0]?.type?.shape?.dimensions;

    // Replace symbolic dims with values from freeDimsOverrides if present
    const resolvedShape = shapeArr.map((dim: any) =>
      typeof dim === 'string' && freeDimsOverrides && freeDimsOverrides[dim] != null
        ? freeDimsOverrides[dim]
        : dim
    );

    inputsCode += `
    const input = builder.input('${name}', { dataType: '${dataType}', shape: [${resolvedShape}] });
    this.inputTensors_['${name}'] = await this.context_.createTensor(
      { dataType: '${dataType}', shape: [${resolvedShape}], writable: true }
    );`;
  }

  let initializersCode = ``;
  const emittedInitializers = new Set<string>();
  const graph = graphModelData?.graph?.[0];
  if (graph && Array.isArray(graph.nodes)) {
    for (const node of graph.nodes) {
      for (const input of node.inputs) {
        if (Array.isArray(input.value)) {
          for (const val of input.value) {
            const initializer = val.initializer;
            if (initializer) {
              let name = initializer.name;
              if (name === undefined || null) {
                name = val.name;
              }
              name = getNonEmptyStringAroundNewline(name);
              const varName = toJsVarName(name);
              if (emittedInitializers.has(varName)) continue; // Skip if already emitted

              const dataType = initializer.type.dataType;
              const shape = initializer.type.shape.dimensions;
              let weightsDataOffset = weightModelData?.[name]?.dataOffset;
              let weightsByteLength = weightModelData?.[name]?.byteLength;

              // TFLite case
              if (weightsDataOffset === undefined || weightsByteLength === undefined) {
                const weightInfo = findWeightNodeByName(weightModelData, name)
                if (weightInfo) {
                  weightsDataOffset = weightInfo.dataOffset;
                  weightsByteLength = weightInfo.byteLength;
                }
              }

              if (initializer?.encoding === '<') {
                const constantBuffer = `new ${getTypedArrayName(dataType)}(weights_array_buffer, ${weightsDataOffset}, ${weightsByteLength} / ${getTypedArrayName(dataType)}.BYTES_PER_ELEMENT)`
                initializersCode += `
    const ${varName} = builder.constant(
      { dataType: '${dataType}', shape: [${shape}] },
      ${constantBuffer}
    );
    `;
                emittedInitializers.add(varName);
              } else if (
                initializer?.encoding === '|' &&
                Array.isArray(initializer?.type.shape.dimensions)
              ) {
                const dims = initializer.type.shape.dimensions;
                const valueArr = Object.keys(initializer.values)
                  .sort((a, b) => Number(a) - Number(b))
                  .map(k => initializer.values[k]);
                const typedArrayCtor = getTypedArrayName(initializer.type.dataType);
                const shapeStr = JSON.stringify(dims);
                initializersCode += `
    const ${varName} = builder.constant(
      { dataType: '${initializer.type.dataType}', shape: ${shapeStr} },
      new ${typedArrayCtor}([${valueArr.join(', ')}])
    );
`;
                emittedInitializers.add(varName);
              }
            }
          }
        }
      }
    }
  }

  let operatorsCode = ``;
  if (graph && Array.isArray(graph.nodes)) {
    for (const node of graph.nodes) {
      const type = node.type.name || '';
      const handler = opHandlers[type];
      if (handler) {
        const jsCode = handler(node, toJsVarName, { nhwc });
        operatorsCode += `    ${jsCode}\n`;
      } else {
        const nodeName = node.name || '';
        const nodeIdentifier = node.identifier || '';
        const nodeNameStr = nodeName ? `· node: ${nodeName} ` : '';
        const nodeIdentifierStr = nodeIdentifier ? `· identifier: ${nodeIdentifier}` : '';
        operatorsCode += `
    // Please file bug into https://github.com/ibelem/webnn-code-generator/issues
    // [Todo] Unsupported op: ${type} ${nodeNameStr}${nodeIdentifierStr}\n`;
      }
    }
  }

  // Build graph with output operands and tensors
  let buildGraphCode = '';
  if (outputs && outputs.length > 0) {
    // Collect output variable names and output names
    const outputVars = outputs.map((output: any) => toJsVarName(getNonEmptyStringAroundNewline(output.value[0]?.name)));
    const outputNames = outputs.map((output: any) => getNonEmptyStringAroundNewline(output.value[0]?.name));

    if (outputVars.length === 1) {
      buildGraphCode += `
    this.graph_ = await builder.build({ '${outputNames[0]}': ${outputVars[0]} });`;
    } else {
      const outputsMap = outputNames.map((name: string, i: number) => `'${name}': ${outputVars[i]}`).join(', ');
      buildGraphCode += `
    this.graph_ = await builder.build({ ${outputsMap} });`;
    }
  }

  let outputsCode = ``;
  for (const output of outputs) {
    const name = getNonEmptyStringAroundNewline(output.value[0]?.name);
    const dataType = output.value[0]?.type?.dataType;
    const shapeArr = output.value[0]?.type?.shape?.dimensions;

    // Replace symbolic dimensions with values from freeDimsOverrides if present
    const resolvedShape = shapeArr.map((dim: any) =>
      typeof dim === 'string' && freeDimsOverrides && freeDimsOverrides[dim] != null
        ? freeDimsOverrides[dim]
        : dim
    );

    outputsCode += `
    this.outputTensors_['${name}'] = await this.context_.createTensor(
      { dataType: '${dataType}', shape: [${resolvedShape}], readable: true }
    );`;
  }

  const binFile = nhwc ? 'weights_nhwc.bin' : 'weights_nchw.bin';
  return `
  async build(options) {
    // Load weights ArrayBuffer from .bin file
    async function loadWeightsArrayBuffer() {
      const binFile = '${binFile}';
      const response = await fetch(binFile);
      if (!response.ok) {
          throw new Error('Failed to fetch weights: ' + response.statusText);
      }
      return await response.arrayBuffer();
    }

    const weights_array_buffer = await loadWeightsArrayBuffer();

    this.context_ = await navigator.ml.createContext(options);
    const builder = new MLGraphBuilder(this.context_);

    // Create graph input operands and tensors
    ${inputsCode}

    // Create graph constant operands
    ${initializersCode}

    // Create graph operators
    ${operatorsCode}

    // Build graph with all outputs
    ${buildGraphCode}

    // Create output tensors
    ${outputsCode}
  }`;
}

export function generateJS() {
  const name = modelName();
  let freeDimsOverridesStr = '';
  if (hasKeysandNumberValues(freeDimsOverrides)) {
    const freeDimsString = Object.entries(freeDimsOverrides)
      .map(([key, value]) => `${key}: ${value}`)
      .join(', ');
    freeDimsOverridesStr = `
  // Set freeDimensionOverrides globally for symbolic dimensions
  // ${freeDimsString}
`;
  }

  // Todo: NCHW, NHWC layouts for BatchNormalization, InstanceNormalization, Conv, ConvInteger, 
  // QLinearConv, ConvTranspose, AveragePool, LpPool, MaxPool, MaxUnpool, GlobalAveragePool, 
  // GlobalLpPool, GlobalMaxPool, LRN, GridSample, DepthToSpace, SpaceToDepth

  // NCHW version
  const nchwClass = `
export class ${name}Nchw {
${freeDimsOverridesStr}${constructorCode()}
${buildCodeWithLayout(false)}
${runCode()}
}`;

  // NHWC version
  const nhwcClass = `
export class ${name}Nhwc {
${freeDimsOverridesStr}${constructorCode()}
${buildCodeWithLayout(true)}
${runCode()}
}`;

  return {
    nchw: `// WebNN Code Generator (NCHW)\n${nchwClass}`,
    nhwc: `// WebNN Code Generator (NHWC)\n${nhwcClass}`
  };
}

export function downloadJS() {
  const jsFiles = generateJS();
  downloadFile(modelName() + '_nchw.js', 'application/javascript', jsFiles.nchw);
  downloadFile(modelName() + '_nhwc.js', 'application/javascript', jsFiles.nhwc);
}

export function generateHTML() {
  const name = modelName();

  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Test ${name} in nchw and nhwc layouts</title>
</head>
<body>
  <h1>Test ${name}.js in nchw and nhwc layouts</h1>
  <button id="run-btn">Build & Run Model</button>
  <label for="deviceType">Device:</label>
  <select id="deviceType">
    <option value="cpu">CPU</option>
    <option value="gpu" selected>GPU</option>
    <option value="npu">NPU</option>
  </select>
  <label for="numRuns">#Runs:</label>
  <input type="number" id="numRuns" value="1" min="1" style="width: 4em;">
  <pre id="output"></pre>
  <script type="module">
    import { ${name}Nchw } from './${name}_nchw.js';
    import { ${name}Nhwc } from './${name}_nhwc.js';

    document.getElementById('run-btn').onclick = async () => {
      const output = document.getElementById('output');
      output.textContent = 'Building model...\\n';
      try {
        const deviceType = document.getElementById('deviceType').value || 'gpu';
        const layout = context.opSupportLimits().preferredInputLayout;
        let model;
        if (layout === 'nchw') {
            model = new ${name}Nchw();
        } else {
            model = new ${name}Nhwc();
        }
        const t0 = performance.now();
        await model.build({ deviceType: deviceType });
        const t1 = performance.now();
        output.textContent += \`Model built successfully. Build latency: \${(t1 - t0).toFixed(2)} ms\\n\`;

        // Output input tensor info
        output.textContent += '\\nInput tensors:\\n';
        for (const name in model.inputTensors_) {
          const tensor = model.inputTensors_[name];
          output.textContent += \`  \${name}: shape=[\${tensor.shape}], dataType=\${tensor.dataType}\\n\`;
        }

        // Output output tensor info
        output.textContent += '\\nOutput tensors:\\n';
        for (const name in model.outputTensors_) {
          const tensor = model.outputTensors_[name];
          output.textContent += \`  \${name}: shape=[\${tensor.shape}], dataType=\${tensor.dataType}\\n\`;
        }
        output.textContent += '\\n';

        // Prepare dummy input data for testing (random values)
        const inputs = {};
        for (const name in model.inputTensors_) {
          const tensor = model.inputTensors_[name];
          let typedArrayCtor = Float32Array;
          switch (tensor.dataType) {
            case 'float32': typedArrayCtor = Float32Array; break;
            case 'uint8': typedArrayCtor = Uint8Array; break;
            case 'int8': typedArrayCtor = Int8Array; break;
            case 'uint16': typedArrayCtor = Uint16Array; break;
            case 'int16': typedArrayCtor = Int16Array; break;
            case 'int32': typedArrayCtor = Int32Array; break;
            case 'int64': typedArrayCtor = BigInt64Array; break;
            case 'float16': typedArrayCtor = Float16Array; break;
            case 'float64': typedArrayCtor = Float64Array; break;
            case 'uint32': typedArrayCtor = Uint32Array; break;
            case 'uint64': typedArrayCtor = BigUint64Array; break;
            default: throw new Error(\`Unhandled input dataType: \${tensor.dataType}\`);
          }
          const size = tensor.shape.reduce((a, b) => a * b, 1);
          const arr = new typedArrayCtor(size);
          // Fill with random values
          if (typedArrayCtor === Float32Array || typedArrayCtor === Float64Array) {
            for (let i = 0; i < size; ++i) arr[i] = Math.random();
          } else if (typedArrayCtor.BYTES_PER_ELEMENT === 8) {
            // BigInt64Array/BigUint64Array
            for (let i = 0; i < size; ++i) arr[i] = BigInt(Math.floor(Math.random() * 100));
          } else {
            for (let i = 0; i < size; ++i) arr[i] = Math.floor(Math.random() * 100);
          }
          inputs[name] = arr;
        }

        output.textContent += 'Running inference...\\n';
        // Get number of runs from input
        let numRuns = parseInt(document.getElementById('numRuns').value) || 1;
        if (numRuns < 1) numRuns = 1;
        // Time model.run and print median inference latency
        const latencies = [];
        let results = null;
        for (let i = 0; i < numRuns; ++i) {
          const t0 = performance.now();
          results = await model.run(inputs);
          const t1 = performance.now();
          latencies.push(t1 - t0);
        }
        latencies.sort((a, b) => a - b);
        const median = latencies[Math.floor(latencies.length / 2)];
        output.textContent += \`Median inference latency (\${numRuns} runs): \${median.toFixed(2)} ms\\n\`;
        output.textContent += '\\n';
        output.textContent += 'Inference results:\\n' + JSON.stringify(results, null, 2);
      } catch (e) {
        output.textContent += 'Error: ' + e;
      }
    };
  </script>
</body>
</html>
`;
}

export function downloadHTML() {
  downloadFile('webnn.html', 'text/html', generateHTML());
}