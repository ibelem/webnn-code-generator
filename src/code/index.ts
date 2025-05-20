// import { getModelState } from './ui';
import { modelName, getTypedArrayName, toJsVarName } from '../utils';
import { getModelState } from '../ui';
import { add_js } from './operation/add';
import { averagePool2d_js } from './operation/averagePool2d';
import { clamp_js } from './operation/clamp';
import { conv2d_js } from './operation/conv2d';
import { gemm_js } from './operation/gemm';
import { reshape_js } from './operation/reshape';

const opHandlers: Record<string, (node: any, toJsVarName: (name: string) => string, initializers?: any[]) => string> = {
  Add: add_js,
  Clip: clamp_js,
  Conv: conv2d_js,
  Gemm: gemm_js,
  GlobalAveragePool: averagePool2d_js,
  Reshape: reshape_js,
  // Add other op handlers here, e.g. Relu: relu_js, etc.
};

function constructorCode() {
  return `
  constructor() {
    this.context_ = null;
    this.graph_ = null;
    this.inputTensors_ = {};
    this.outputTensors_ = {};
  }`;
}

function buildCode() {
  const { graphModelData, weightModelData } = getModelState();
  const inputs = graphModelData?.graph?.[0].inputs;
  const outputs = graphModelData?.graph?.[0].outputs;

  let inputsCode = ``;
  for(const input of inputs) {
    const name = input.value[0]?.name;
    const dataType = input.value[0]?.type?.dataType;
    const shape = input.value[0]?.type?.shape?.dimensions;
    inputsCode += `
    const input = builder.input('${name}', { dataType: '${dataType}', shape: [${shape}] });
    this.inputTensors_['${name}'] = await this.context_.createTensor(
      { dataType: '${dataType}', shape: [${shape}], writable: true }
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
              const name = initializer.name;
              const varName = name.replaceAll('.', '_');
              if (emittedInitializers.has(varName)) continue; // Skip if already emitted

              const dataType = initializer.type.dataType;
              const shape = initializer.type.shape.dimensions;
              const weightsDataOffset = weightModelData?.[name]?.dataOffset;
              const weightsByteLength = weightModelData?.[name]?.byteLength;

              if (initializer?.encoding === '<') {
                initializersCode += `
    const var_${varName} = builder.constant(
      { dataType: '${dataType}', shape: [${shape}] },
      new ${getTypedArrayName(dataType)}(weights_array_buffer, ${weightsDataOffset}, ${weightsByteLength} / ${getTypedArrayName(dataType)}.BYTES_PER_ELEMENT)
    );
    `;
                emittedInitializers.add(varName);
              } else if (
                initializer?.encoding === '|' &&
                Array.isArray(initializer?.type.shape.dimensions) &&
                initializer?.type.shape.dimensions.length === 1 &&
                initializer?.type.shape.dimensions[0] === 1
              ) {
                initializersCode += `
    const var_${varName} = ${initializer?.values['0']};
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
        const jsCode = handler(node, toJsVarName);
        operatorsCode += `    ${jsCode}\n`;
      } else {
        const nodeName = node.name || '';
        operatorsCode += `
    // Unsupported op: ${type} (node: ${nodeName})\n`;
      }
    }
  }

  // Build graph with output operands and tensors
  let buildGraphCode = '';
  if (outputs && outputs.length > 0) {
    // Collect output variable names and output names
    const outputVars = outputs.map((output: any) => toJsVarName(output.value[0]?.name));
    const outputNames = outputs.map((output: any) => output.value[0]?.name);

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
  for(const output of outputs) {
    const name = output.value[0]?.name;
    const dataType = output.value[0]?.type?.dataType;
    const shape = output.value[0]?.type?.shape?.dimensions;
    outputsCode += `
    this.outputTensors_['${name}'] = await this.context_.createTensor(
      { dataType: '${dataType}', shape: [${shape}], writable: true }
    );`;
  }

  return `
  async build(options) {
    // Load weights ArrayBuffer from .bin file
    async function loadWeightsArrayBuffer() {
      const response = await fetch('model.bin');
      if (!response.ok) {
          throw new Error('Failed to fetch weights: ' + response.statusText);
      }
      return await response.arrayBuffer();
    }

    const weightsArrayBuffer = await loadWeightsArrayBuffer();

    this.context_ = await navigator.ml.createContext(contextOptions);
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
        case 'float64': typedArrayCtor = Float64Array; break;
        case 'float16': typedArrayCtor = Float16Array; break;
        case 'uint32': typedArrayCtor = Uint32Array; break;
        case 'uint64': typedArrayCtor = BigUint64Array; break;
        default: throw new Error('Unhandled tensor dataType: ' + tensor.dataType);
      }
      outputs[name] = new typedArrayCtor(buffer);
    }
    return outputs;
  }`;
}

export function generateJS() {
  const name = modelName();
  return `// WebNN Code Generator
// Todo: setFreeDims globally
// Todo: NCHW, NHWC layouts

export class ${name} {
${constructorCode()}
${buildCode()}
${runCode()}
}`;
}

export function generateHTML() {
  const name = modelName();
  return `export class ${name} {
${constructorCode()}
${buildCode()}
${runCode()}
}`;
};