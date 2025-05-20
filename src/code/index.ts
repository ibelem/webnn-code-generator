// import { getModelState } from './ui';
import { modelName, getTypedArrayName } from '../utils';
import { getModelState } from '../ui';

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
  const graph = graphModelData?.graph?.[0];
  if (graph && Array.isArray(graph.nodes)) {
    for (const node of graph.nodes) {
      for (const input of node.inputs) {
        if (Array.isArray(input.value)) {
          for (const val of input.value) {
            const initializer = val.initializer;
            if (initializer) {
              const name = initializer.name;
              const dataType = initializer.type.dataType;
              const shape = initializer.type.shape.dimensions;

              const weightsDataOffset = weightModelData?.[name]?.dataOffset;
              const weightsByteLength = weightModelData?.[name]?.byteLength;
              const varName = name.replaceAll('.', '_');

              if(initializer?.encoding === '<') {
                initializersCode += `
    const var_${varName} = builder.constant(
        { dataType: '${dataType}', shape: [${shape}] },
        new ${getTypedArrayName(dataType)}(weights_array_buffer, ${weightsDataOffset}, ${weightsByteLength} / ${getTypedArrayName(dataType)}.BYTES_PER_ELEMENT)
    );`;
              } else if (
                initializer?.encoding === '|' &&
                Array.isArray(initializer?.type.shape.dimensions) &&
                initializer?.type.shape.dimensions.length === 1 &&
                initializer?.type.shape.dimensions[0] === 1
              ) {
                initializersCode += `
    const var_${varName} = ${initializer?.values['0']};`;
              }
            }
          }
        }
      }
    }
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
        case 'float64': typedArrayCtor = Float64Array; break;
        case 'int32': typedArrayCtor = Int32Array; break;
        case 'uint8': typedArrayCtor = Uint8Array; break;
        case 'int8': typedArrayCtor = Int8Array; break;
        case 'uint16': typedArrayCtor = Uint16Array; break;
        case 'int16': typedArrayCtor = Int16Array; break;
        case 'uint32': typedArrayCtor = Uint32Array; break;
        case 'int64': typedArrayCtor = BigInt64Array; break;
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
  return `export class ${name} {
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
}