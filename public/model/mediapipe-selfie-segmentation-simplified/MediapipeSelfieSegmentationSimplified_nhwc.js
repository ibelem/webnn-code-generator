// WebNN Code Generator (NHWC)

export class MediapipeSelfieSegmentationSimplifiedNhwc {

  constructor() {
    this.context_ = null;
    this.graph_ = null;
    this.inputTensors_ = {};
    this.outputTensors_ = {};
  }

  async build(options) {
    // Load weights ArrayBuffer from .bin file
    async function loadWeightsArrayBuffer() {
      const binFile = 'weights_nhwc.bin';
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
    
    const pixel_values = builder.transpose(
      builder.input('pixel_values', { dataType: 'float32', shape: [1,3,256,256] }),
      { permutation: [0, 2, 3, 1] }
    );

    this.inputTensors_['pixel_values'] = await this.context_.createTensor(
      { dataType: 'float32', shape: [1,3,256,256], writable: true }
    );

    // Initializers, create graph constant operands
    
    const var_conv1_weight = builder.constant(
      { dataType: 'float32', shape: [16,3,3,3] },
      new Float32Array(weights_array_buffer, 0, 1728 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_conv1_bias = builder.constant(
      { dataType: 'float32', shape: [16] },
      new Float32Array(weights_array_buffer, 1728, 64 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_conv2_1_weight = builder.constant(
      { dataType: 'float32', shape: [16,1,1,16] },
      new Float32Array(weights_array_buffer, 1792, 1024 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_conv2_1_bias = builder.constant(
      { dataType: 'float32', shape: [16] },
      new Float32Array(weights_array_buffer, 2816, 64 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_conv2_2_weight = builder.constant(
      { dataType: 'float32', shape: [1,3,3,16] },
      new Float32Array(weights_array_buffer, 2880, 576 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_conv2_2_bias = builder.constant(
      { dataType: 'float32', shape: [16] },
      new Float32Array(weights_array_buffer, 3456, 64 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_conv3_1_weight = builder.constant(
      { dataType: 'float32', shape: [8,1,1,16] },
      new Float32Array(weights_array_buffer, 3520, 512 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_conv3_1_bias = builder.constant(
      { dataType: 'float32', shape: [8] },
      new Float32Array(weights_array_buffer, 4032, 32 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_conv3_2_weight = builder.constant(
      { dataType: 'float32', shape: [16,1,1,8] },
      new Float32Array(weights_array_buffer, 4064, 512 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_conv3_2_bias = builder.constant(
      { dataType: 'float32', shape: [16] },
      new Float32Array(weights_array_buffer, 4576, 64 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_conv3_3_weight = builder.constant(
      { dataType: 'float32', shape: [16,1,1,16] },
      new Float32Array(weights_array_buffer, 4640, 1024 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_conv3_3_bias = builder.constant(
      { dataType: 'float32', shape: [16] },
      new Float32Array(weights_array_buffer, 5664, 64 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_conv4_1_weight = builder.constant(
      { dataType: 'float32', shape: [72,1,1,16] },
      new Float32Array(weights_array_buffer, 5728, 4608 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_conv4_1_bias = builder.constant(
      { dataType: 'float32', shape: [72] },
      new Float32Array(weights_array_buffer, 10336, 288 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_conv4_2_weight = builder.constant(
      { dataType: 'float32', shape: [1,3,3,72] },
      new Float32Array(weights_array_buffer, 10624, 2592 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_conv4_2_bias = builder.constant(
      { dataType: 'float32', shape: [72] },
      new Float32Array(weights_array_buffer, 13216, 288 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_conv4_3_weight = builder.constant(
      { dataType: 'float32', shape: [24,1,1,72] },
      new Float32Array(weights_array_buffer, 13504, 6912 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_conv4_3_bias = builder.constant(
      { dataType: 'float32', shape: [24] },
      new Float32Array(weights_array_buffer, 20416, 96 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_conv5_1_weight = builder.constant(
      { dataType: 'float32', shape: [88,1,1,24] },
      new Float32Array(weights_array_buffer, 20512, 8448 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_conv5_1_bias = builder.constant(
      { dataType: 'float32', shape: [88] },
      new Float32Array(weights_array_buffer, 28960, 352 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_conv5_2_weight = builder.constant(
      { dataType: 'float32', shape: [1,3,3,88] },
      new Float32Array(weights_array_buffer, 29312, 3168 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_conv5_2_bias = builder.constant(
      { dataType: 'float32', shape: [88] },
      new Float32Array(weights_array_buffer, 32480, 352 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_conv5_3_weight = builder.constant(
      { dataType: 'float32', shape: [24,1,1,88] },
      new Float32Array(weights_array_buffer, 32832, 8448 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_conv5_3_bias = builder.constant(
      { dataType: 'float32', shape: [24] },
      new Float32Array(weights_array_buffer, 41280, 96 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_conv6_1_weight = builder.constant(
      { dataType: 'float32', shape: [96,1,1,24] },
      new Float32Array(weights_array_buffer, 41376, 9216 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_conv6_1_bias = builder.constant(
      { dataType: 'float32', shape: [96] },
      new Float32Array(weights_array_buffer, 50592, 384 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_conv6_2_weight = builder.constant(
      { dataType: 'float32', shape: [1,5,5,96] },
      new Float32Array(weights_array_buffer, 50976, 9600 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_conv6_2_bias = builder.constant(
      { dataType: 'float32', shape: [96] },
      new Float32Array(weights_array_buffer, 60576, 384 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_sub_block1_conv1_weight = builder.constant(
      { dataType: 'float32', shape: [24,1,1,96] },
      new Float32Array(weights_array_buffer, 60960, 9216 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_sub_block1_conv1_bias = builder.constant(
      { dataType: 'float32', shape: [24] },
      new Float32Array(weights_array_buffer, 70176, 96 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_sub_block1_conv2_weight = builder.constant(
      { dataType: 'float32', shape: [96,1,1,24] },
      new Float32Array(weights_array_buffer, 70272, 9216 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_sub_block1_conv2_bias = builder.constant(
      { dataType: 'float32', shape: [96] },
      new Float32Array(weights_array_buffer, 79488, 384 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_sub_block1_conv3_weight = builder.constant(
      { dataType: 'float32', shape: [32,1,1,96] },
      new Float32Array(weights_array_buffer, 79872, 12288 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_sub_block1_conv3_bias = builder.constant(
      { dataType: 'float32', shape: [32] },
      new Float32Array(weights_array_buffer, 92160, 128 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_block1_conv1_weight = builder.constant(
      { dataType: 'float32', shape: [128,1,1,32] },
      new Float32Array(weights_array_buffer, 92288, 16384 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_block1_conv1_bias = builder.constant(
      { dataType: 'float32', shape: [128] },
      new Float32Array(weights_array_buffer, 108672, 512 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_block1_conv2_weight = builder.constant(
      { dataType: 'float32', shape: [1,5,5,128] },
      new Float32Array(weights_array_buffer, 109184, 12800 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_block1_conv2_bias = builder.constant(
      { dataType: 'float32', shape: [128] },
      new Float32Array(weights_array_buffer, 121984, 512 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_block1_sub_block_conv1_weight = builder.constant(
      { dataType: 'float32', shape: [32,1,1,128] },
      new Float32Array(weights_array_buffer, 122496, 16384 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_block1_sub_block_conv1_bias = builder.constant(
      { dataType: 'float32', shape: [32] },
      new Float32Array(weights_array_buffer, 138880, 128 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_block1_sub_block_conv2_weight = builder.constant(
      { dataType: 'float32', shape: [128,1,1,32] },
      new Float32Array(weights_array_buffer, 139008, 16384 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_block1_sub_block_conv2_bias = builder.constant(
      { dataType: 'float32', shape: [128] },
      new Float32Array(weights_array_buffer, 155392, 512 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_block1_sub_block_conv3_weight = builder.constant(
      { dataType: 'float32', shape: [32,1,1,128] },
      new Float32Array(weights_array_buffer, 155904, 16384 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_block1_sub_block_conv3_bias = builder.constant(
      { dataType: 'float32', shape: [32] },
      new Float32Array(weights_array_buffer, 172288, 128 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_block2_conv1_weight = builder.constant(
      { dataType: 'float32', shape: [128,1,1,32] },
      new Float32Array(weights_array_buffer, 172416, 16384 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_block2_conv1_bias = builder.constant(
      { dataType: 'float32', shape: [128] },
      new Float32Array(weights_array_buffer, 188800, 512 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_block2_conv2_weight = builder.constant(
      { dataType: 'float32', shape: [1,5,5,128] },
      new Float32Array(weights_array_buffer, 189312, 12800 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_block2_conv2_bias = builder.constant(
      { dataType: 'float32', shape: [128] },
      new Float32Array(weights_array_buffer, 202112, 512 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_block2_sub_block_conv1_weight = builder.constant(
      { dataType: 'float32', shape: [32,1,1,128] },
      new Float32Array(weights_array_buffer, 202624, 16384 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_block2_sub_block_conv1_bias = builder.constant(
      { dataType: 'float32', shape: [32] },
      new Float32Array(weights_array_buffer, 219008, 128 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_block2_sub_block_conv2_weight = builder.constant(
      { dataType: 'float32', shape: [128,1,1,32] },
      new Float32Array(weights_array_buffer, 219136, 16384 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_block2_sub_block_conv2_bias = builder.constant(
      { dataType: 'float32', shape: [128] },
      new Float32Array(weights_array_buffer, 235520, 512 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_block2_sub_block_conv3_weight = builder.constant(
      { dataType: 'float32', shape: [32,1,1,128] },
      new Float32Array(weights_array_buffer, 236032, 16384 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_block2_sub_block_conv3_bias = builder.constant(
      { dataType: 'float32', shape: [32] },
      new Float32Array(weights_array_buffer, 252416, 128 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_block3_conv1_weight = builder.constant(
      { dataType: 'float32', shape: [96,1,1,32] },
      new Float32Array(weights_array_buffer, 252544, 12288 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_block3_conv1_bias = builder.constant(
      { dataType: 'float32', shape: [96] },
      new Float32Array(weights_array_buffer, 264832, 384 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_block3_conv2_weight = builder.constant(
      { dataType: 'float32', shape: [1,5,5,96] },
      new Float32Array(weights_array_buffer, 265216, 9600 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_block3_conv2_bias = builder.constant(
      { dataType: 'float32', shape: [96] },
      new Float32Array(weights_array_buffer, 274816, 384 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_block3_sub_block_conv1_weight = builder.constant(
      { dataType: 'float32', shape: [24,1,1,96] },
      new Float32Array(weights_array_buffer, 275200, 9216 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_block3_sub_block_conv1_bias = builder.constant(
      { dataType: 'float32', shape: [24] },
      new Float32Array(weights_array_buffer, 284416, 96 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_block3_sub_block_conv2_weight = builder.constant(
      { dataType: 'float32', shape: [96,1,1,24] },
      new Float32Array(weights_array_buffer, 284512, 9216 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_block3_sub_block_conv2_bias = builder.constant(
      { dataType: 'float32', shape: [96] },
      new Float32Array(weights_array_buffer, 293728, 384 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_block3_sub_block_conv3_weight = builder.constant(
      { dataType: 'float32', shape: [32,1,1,96] },
      new Float32Array(weights_array_buffer, 294112, 12288 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_block3_sub_block_conv3_bias = builder.constant(
      { dataType: 'float32', shape: [32] },
      new Float32Array(weights_array_buffer, 306400, 128 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_block4_conv1_weight = builder.constant(
      { dataType: 'float32', shape: [96,1,1,32] },
      new Float32Array(weights_array_buffer, 306528, 12288 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_block4_conv1_bias = builder.constant(
      { dataType: 'float32', shape: [96] },
      new Float32Array(weights_array_buffer, 318816, 384 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_block4_conv2_weight = builder.constant(
      { dataType: 'float32', shape: [1,5,5,96] },
      new Float32Array(weights_array_buffer, 319200, 9600 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_block4_conv2_bias = builder.constant(
      { dataType: 'float32', shape: [96] },
      new Float32Array(weights_array_buffer, 328800, 384 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_block4_sub_block_conv1_weight = builder.constant(
      { dataType: 'float32', shape: [24,1,1,96] },
      new Float32Array(weights_array_buffer, 329184, 9216 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_block4_sub_block_conv1_bias = builder.constant(
      { dataType: 'float32', shape: [24] },
      new Float32Array(weights_array_buffer, 338400, 96 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_block4_sub_block_conv2_weight = builder.constant(
      { dataType: 'float32', shape: [96,1,1,24] },
      new Float32Array(weights_array_buffer, 338496, 9216 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_block4_sub_block_conv2_bias = builder.constant(
      { dataType: 'float32', shape: [96] },
      new Float32Array(weights_array_buffer, 347712, 384 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_block4_sub_block_conv3_weight = builder.constant(
      { dataType: 'float32', shape: [32,1,1,96] },
      new Float32Array(weights_array_buffer, 348096, 12288 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_block4_sub_block_conv3_bias = builder.constant(
      { dataType: 'float32', shape: [32] },
      new Float32Array(weights_array_buffer, 360384, 128 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_conv7_weight = builder.constant(
      { dataType: 'float32', shape: [128,1,1,32] },
      new Float32Array(weights_array_buffer, 360512, 16384 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_conv7_bias = builder.constant(
      { dataType: 'float32', shape: [128] },
      new Float32Array(weights_array_buffer, 376896, 512 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_conv8_weight = builder.constant(
      { dataType: 'float32', shape: [128,1,1,32] },
      new Float32Array(weights_array_buffer, 377408, 16384 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_conv8_bias = builder.constant(
      { dataType: 'float32', shape: [128] },
      new Float32Array(weights_array_buffer, 393792, 512 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_final_block1_conv1_weight = builder.constant(
      { dataType: 'float32', shape: [24,1,1,128] },
      new Float32Array(weights_array_buffer, 394336, 12288 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_final_block1_conv1_bias = builder.constant(
      { dataType: 'float32', shape: [24] },
      new Float32Array(weights_array_buffer, 406624, 96 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_final_block1_conv2_weight = builder.constant(
      { dataType: 'float32', shape: [24,1,1,24] },
      new Float32Array(weights_array_buffer, 406720, 2304 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_final_block1_conv2_bias = builder.constant(
      { dataType: 'float32', shape: [24] },
      new Float32Array(weights_array_buffer, 409024, 96 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_final_block1_conv3_weight = builder.constant(
      { dataType: 'float32', shape: [24,1,1,24] },
      new Float32Array(weights_array_buffer, 409120, 2304 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_final_block1_conv3_bias = builder.constant(
      { dataType: 'float32', shape: [24] },
      new Float32Array(weights_array_buffer, 411424, 96 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_final_block1_conv4_weight = builder.constant(
      { dataType: 'float32', shape: [24,1,1,24] },
      new Float32Array(weights_array_buffer, 411520, 2304 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_final_block1_conv4_bias = builder.constant(
      { dataType: 'float32', shape: [24] },
      new Float32Array(weights_array_buffer, 413824, 96 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_final_block1_conv5_weight = builder.constant(
      { dataType: 'float32', shape: [1,3,3,24] },
      new Float32Array(weights_array_buffer, 413920, 864 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_final_block1_conv5_bias = builder.constant(
      { dataType: 'float32', shape: [24] },
      new Float32Array(weights_array_buffer, 414784, 96 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_final_block2_conv1_weight = builder.constant(
      { dataType: 'float32', shape: [16,1,1,24] },
      new Float32Array(weights_array_buffer, 414912, 1536 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_final_block2_conv1_bias = builder.constant(
      { dataType: 'float32', shape: [16] },
      new Float32Array(weights_array_buffer, 416448, 64 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_final_block2_conv2_weight = builder.constant(
      { dataType: 'float32', shape: [16,1,1,16] },
      new Float32Array(weights_array_buffer, 416512, 1024 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_final_block2_conv2_bias = builder.constant(
      { dataType: 'float32', shape: [16] },
      new Float32Array(weights_array_buffer, 417536, 64 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_final_block2_conv3_weight = builder.constant(
      { dataType: 'float32', shape: [16,1,1,16] },
      new Float32Array(weights_array_buffer, 417600, 1024 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_final_block2_conv3_bias = builder.constant(
      { dataType: 'float32', shape: [16] },
      new Float32Array(weights_array_buffer, 418624, 64 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_final_block2_conv4_weight = builder.constant(
      { dataType: 'float32', shape: [16,1,1,16] },
      new Float32Array(weights_array_buffer, 418688, 1024 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_final_block2_conv4_bias = builder.constant(
      { dataType: 'float32', shape: [16] },
      new Float32Array(weights_array_buffer, 419712, 64 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_final_block2_conv5_weight = builder.constant(
      { dataType: 'float32', shape: [1,3,3,16] },
      new Float32Array(weights_array_buffer, 419776, 576 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_final_block2_conv5_bias = builder.constant(
      { dataType: 'float32', shape: [16] },
      new Float32Array(weights_array_buffer, 420352, 64 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_final_block3_conv1_weight = builder.constant(
      { dataType: 'float32', shape: [16,1,1,16] },
      new Float32Array(weights_array_buffer, 420448, 1024 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_final_block3_conv1_bias = builder.constant(
      { dataType: 'float32', shape: [16] },
      new Float32Array(weights_array_buffer, 421472, 64 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_final_block3_conv2_weight = builder.constant(
      { dataType: 'float32', shape: [16,1,1,16] },
      new Float32Array(weights_array_buffer, 421536, 1024 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_final_block3_conv2_bias = builder.constant(
      { dataType: 'float32', shape: [16] },
      new Float32Array(weights_array_buffer, 422560, 64 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_final_block3_conv3_weight = builder.constant(
      { dataType: 'float32', shape: [16,1,1,16] },
      new Float32Array(weights_array_buffer, 422624, 1024 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_final_block3_conv3_bias = builder.constant(
      { dataType: 'float32', shape: [16] },
      new Float32Array(weights_array_buffer, 423648, 64 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_final_block3_conv4_weight = builder.constant(
      { dataType: 'float32', shape: [16,1,1,16] },
      new Float32Array(weights_array_buffer, 423712, 1024 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_final_block3_conv4_bias = builder.constant(
      { dataType: 'float32', shape: [16] },
      new Float32Array(weights_array_buffer, 424736, 64 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_final_block3_conv5_weight = builder.constant(
      { dataType: 'float32', shape: [1,3,3,16] },
      new Float32Array(weights_array_buffer, 424800, 576 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_final_block3_conv5_bias = builder.constant(
      { dataType: 'float32', shape: [16] },
      new Float32Array(weights_array_buffer, 425376, 64 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_conv_transpose_weight = builder.constant(
      { dataType: 'float32', shape: [1,2,2,16] },
      new Float32Array(weights_array_buffer, 425440, 256 / Float32Array.BYTES_PER_ELEMENT)
    );

    const var_conv_transpose_bias = builder.constant(
      { dataType: 'float32', shape: [1] },
      new Float32Array(weights_array_buffer, 425696, 4 / Float32Array.BYTES_PER_ELEMENT)
    );

    // Create graph operators
        
    const var__conv1_conv_output_0 = builder.conv2d(
      pixel_values, var_conv1_weight,
      {
        strides: [2, 2],
        padding: [0, 1, 0, 1],
        dilations: [1, 1],
        groups: 1,
        bias: var_conv1_bias,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: '/conv1/Conv'
      }
    );
    
    const var__hardswish_output_0 = builder.hardSwish(
      var__conv1_conv_output_0,
      { label: '/HardSwish' }
    );
    
    const var__conv2_1_conv_output_0 = builder.conv2d(
      var__hardswish_output_0, var_conv2_1_weight,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_conv2_1_bias,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: '/conv2_1/Conv'
      }
    );
    
    const var__relu_output_0 = builder.relu(
      var__conv2_1_conv_output_0,
      { label: '/Relu' }
    );
    
    const var__conv2_2_conv_output_0 = builder.conv2d(
      var__relu_output_0, var_conv2_2_weight,
      {
        strides: [2, 2],
        padding: [0, 1, 0, 1],
        dilations: [1, 1],
        groups: 16,
        bias: var_conv2_2_bias,
        filterLayout: 'ihwo',
        inputLayout: 'nhwc',
        label: '/conv2_2/Conv'
      }
    );
    
    const var__relu_1_output_0 = builder.relu(
      var__conv2_2_conv_output_0,
      { label: '/Relu_1' }
    );
    
    const var__reducemean_output_0 = builder.reduceMean(
      var__relu_1_output_0,
      { keepDimensions: true, axes: [1, 2], label: '/ReduceMean' }
    );
    
    const var__conv3_1_conv_output_0 = builder.conv2d(
      var__reducemean_output_0, var_conv3_1_weight,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_conv3_1_bias,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: '/conv3_1/Conv'
      }
    );
    
    const var__relu_2_output_0 = builder.relu(
      var__conv3_1_conv_output_0,
      { label: '/Relu_2' }
    );
    
    const var__conv3_2_conv_output_0 = builder.conv2d(
      var__relu_2_output_0, var_conv3_2_weight,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_conv3_2_bias,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: '/conv3_2/Conv'
      }
    );
    
    const var__sigmoid_output_0 = builder.sigmoid(
      var__conv3_2_conv_output_0,
      { label: '/Sigmoid' }
    );
    
    const var__mul_output_0 = builder.mul(
      var__relu_1_output_0,
      var__sigmoid_output_0,
      { label: '/Mul' }
    );
    
    const var__conv3_3_conv_output_0 = builder.conv2d(
      var__mul_output_0, var_conv3_3_weight,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_conv3_3_bias,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: '/conv3_3/Conv'
      }
    );
    
    const var__conv4_1_conv_output_0 = builder.conv2d(
      var__conv3_3_conv_output_0, var_conv4_1_weight,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_conv4_1_bias,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: '/conv4_1/Conv'
      }
    );
    
    const var__relu_3_output_0 = builder.relu(
      var__conv4_1_conv_output_0,
      { label: '/Relu_3' }
    );
    
    const var__conv4_2_conv_output_0 = builder.conv2d(
      var__relu_3_output_0, var_conv4_2_weight,
      {
        strides: [2, 2],
        padding: [0, 1, 0, 1],
        dilations: [1, 1],
        groups: 72,
        bias: var_conv4_2_bias,
        filterLayout: 'ihwo',
        inputLayout: 'nhwc',
        label: '/conv4_2/Conv'
      }
    );
    
    const var__relu_4_output_0 = builder.relu(
      var__conv4_2_conv_output_0,
      { label: '/Relu_4' }
    );
    
    const var__conv4_3_conv_output_0 = builder.conv2d(
      var__relu_4_output_0, var_conv4_3_weight,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_conv4_3_bias,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: '/conv4_3/Conv'
      }
    );
    
    const var__conv5_1_conv_output_0 = builder.conv2d(
      var__conv4_3_conv_output_0, var_conv5_1_weight,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_conv5_1_bias,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: '/conv5_1/Conv'
      }
    );
    
    const var__relu_5_output_0 = builder.relu(
      var__conv5_1_conv_output_0,
      { label: '/Relu_5' }
    );
    
    const var__conv5_2_conv_output_0 = builder.conv2d(
      var__relu_5_output_0, var_conv5_2_weight,
      {
        strides: [1, 1],
        padding: [1, 1, 1, 1],
        dilations: [1, 1],
        groups: 88,
        bias: var_conv5_2_bias,
        filterLayout: 'ihwo',
        inputLayout: 'nhwc',
        label: '/conv5_2/Conv'
      }
    );
    
    const var__relu_6_output_0 = builder.relu(
      var__conv5_2_conv_output_0,
      { label: '/Relu_6' }
    );
    
    const var__conv5_3_conv_output_0 = builder.conv2d(
      var__relu_6_output_0, var_conv5_3_weight,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_conv5_3_bias,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: '/conv5_3/Conv'
      }
    );
    
    const var__add_output_0 = builder.add(
      var__conv4_3_conv_output_0,
      var__conv5_3_conv_output_0,
      { label: '/Add' }
    );
    
    const var__conv6_1_conv_output_0 = builder.conv2d(
      var__add_output_0, var_conv6_1_weight,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_conv6_1_bias,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: '/conv6_1/Conv'
      }
    );
    
    const var__hardswish_1_output_0 = builder.hardSwish(
      var__conv6_1_conv_output_0,
      { label: '/HardSwish_1' }
    );
    
    const var__conv6_2_conv_output_0 = builder.conv2d(
      var__hardswish_1_output_0, var_conv6_2_weight,
      {
        strides: [2, 2],
        padding: [1, 2, 1, 2],
        dilations: [1, 1],
        groups: 96,
        bias: var_conv6_2_bias,
        filterLayout: 'ihwo',
        inputLayout: 'nhwc',
        label: '/conv6_2/Conv'
      }
    );
    
    const var__sub_block1_hardswish_output_0 = builder.hardSwish(
      var__conv6_2_conv_output_0,
      { label: '/sub_block1/HardSwish' }
    );
    
    const var__sub_block1_reducemean_output_0 = builder.reduceMean(
      var__sub_block1_hardswish_output_0,
      { keepDimensions: true, axes: [1, 2], label: '/sub_block1/ReduceMean' }
    );
    
    const var__sub_block1_conv1_conv_output_0 = builder.conv2d(
      var__sub_block1_reducemean_output_0, var_sub_block1_conv1_weight,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_sub_block1_conv1_bias,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: '/sub_block1/conv1/Conv'
      }
    );
    
    const var__sub_block1_relu_output_0 = builder.relu(
      var__sub_block1_conv1_conv_output_0,
      { label: '/sub_block1/Relu' }
    );
    
    const var__sub_block1_conv2_conv_output_0 = builder.conv2d(
      var__sub_block1_relu_output_0, var_sub_block1_conv2_weight,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_sub_block1_conv2_bias,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: '/sub_block1/conv2/Conv'
      }
    );
    
    const var__sub_block1_sigmoid_output_0 = builder.sigmoid(
      var__sub_block1_conv2_conv_output_0,
      { label: '/sub_block1/Sigmoid' }
    );
    
    const var__sub_block1_mul_output_0 = builder.mul(
      var__sub_block1_hardswish_output_0,
      var__sub_block1_sigmoid_output_0,
      { label: '/sub_block1/Mul' }
    );
    
    const var__sub_block1_conv3_conv_output_0 = builder.conv2d(
      var__sub_block1_mul_output_0, var_sub_block1_conv3_weight,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_sub_block1_conv3_bias,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: '/sub_block1/conv3/Conv'
      }
    );
    
    const var__block1_conv1_conv_output_0 = builder.conv2d(
      var__sub_block1_conv3_conv_output_0, var_block1_conv1_weight,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_block1_conv1_bias,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: '/block1/conv1/Conv'
      }
    );
    
    const var__block1_hardswish_output_0 = builder.hardSwish(
      var__block1_conv1_conv_output_0,
      { label: '/block1/HardSwish' }
    );
    
    const var__block1_conv2_conv_output_0 = builder.conv2d(
      var__block1_hardswish_output_0, var_block1_conv2_weight,
      {
        strides: [1, 1],
        padding: [2, 2, 2, 2],
        dilations: [1, 1],
        groups: 128,
        bias: var_block1_conv2_bias,
        filterLayout: 'ihwo',
        inputLayout: 'nhwc',
        label: '/block1/conv2/Conv'
      }
    );
    
    const var__block1_sub_block_hardswish_output_0 = builder.hardSwish(
      var__block1_conv2_conv_output_0,
      { label: '/block1/sub_block/HardSwish' }
    );
    
    const var__block1_sub_block_reducemean_output_0 = builder.reduceMean(
      var__block1_sub_block_hardswish_output_0,
      { keepDimensions: true, axes: [1, 2], label: '/block1/sub_block/ReduceMean' }
    );
    
    const var__block1_sub_block_conv1_conv_output_0 = builder.conv2d(
      var__block1_sub_block_reducemean_output_0, var_block1_sub_block_conv1_weight,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_block1_sub_block_conv1_bias,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: '/block1/sub_block/conv1/Conv'
      }
    );
    
    const var__block1_sub_block_relu_output_0 = builder.relu(
      var__block1_sub_block_conv1_conv_output_0,
      { label: '/block1/sub_block/Relu' }
    );
    
    const var__block1_sub_block_conv2_conv_output_0 = builder.conv2d(
      var__block1_sub_block_relu_output_0, var_block1_sub_block_conv2_weight,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_block1_sub_block_conv2_bias,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: '/block1/sub_block/conv2/Conv'
      }
    );
    
    const var__block1_sub_block_sigmoid_output_0 = builder.sigmoid(
      var__block1_sub_block_conv2_conv_output_0,
      { label: '/block1/sub_block/Sigmoid' }
    );
    
    const var__block1_sub_block_mul_output_0 = builder.mul(
      var__block1_sub_block_hardswish_output_0,
      var__block1_sub_block_sigmoid_output_0,
      { label: '/block1/sub_block/Mul' }
    );
    
    const var__block1_sub_block_conv3_conv_output_0 = builder.conv2d(
      var__block1_sub_block_mul_output_0, var_block1_sub_block_conv3_weight,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_block1_sub_block_conv3_bias,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: '/block1/sub_block/conv3/Conv'
      }
    );
    
    const var__block1_add_output_0 = builder.add(
      var__sub_block1_conv3_conv_output_0,
      var__block1_sub_block_conv3_conv_output_0,
      { label: '/block1/Add' }
    );
    
    const var__block2_conv1_conv_output_0 = builder.conv2d(
      var__block1_add_output_0, var_block2_conv1_weight,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_block2_conv1_bias,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: '/block2/conv1/Conv'
      }
    );
    
    const var__block2_hardswish_output_0 = builder.hardSwish(
      var__block2_conv1_conv_output_0,
      { label: '/block2/HardSwish' }
    );
    
    const var__block2_conv2_conv_output_0 = builder.conv2d(
      var__block2_hardswish_output_0, var_block2_conv2_weight,
      {
        strides: [1, 1],
        padding: [2, 2, 2, 2],
        dilations: [1, 1],
        groups: 128,
        bias: var_block2_conv2_bias,
        filterLayout: 'ihwo',
        inputLayout: 'nhwc',
        label: '/block2/conv2/Conv'
      }
    );
    
    const var__block2_sub_block_hardswish_output_0 = builder.hardSwish(
      var__block2_conv2_conv_output_0,
      { label: '/block2/sub_block/HardSwish' }
    );
    
    const var__block2_sub_block_reducemean_output_0 = builder.reduceMean(
      var__block2_sub_block_hardswish_output_0,
      { keepDimensions: true, axes: [1, 2], label: '/block2/sub_block/ReduceMean' }
    );
    
    const var__block2_sub_block_conv1_conv_output_0 = builder.conv2d(
      var__block2_sub_block_reducemean_output_0, var_block2_sub_block_conv1_weight,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_block2_sub_block_conv1_bias,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: '/block2/sub_block/conv1/Conv'
      }
    );
    
    const var__block2_sub_block_relu_output_0 = builder.relu(
      var__block2_sub_block_conv1_conv_output_0,
      { label: '/block2/sub_block/Relu' }
    );
    
    const var__block2_sub_block_conv2_conv_output_0 = builder.conv2d(
      var__block2_sub_block_relu_output_0, var_block2_sub_block_conv2_weight,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_block2_sub_block_conv2_bias,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: '/block2/sub_block/conv2/Conv'
      }
    );
    
    const var__block2_sub_block_sigmoid_output_0 = builder.sigmoid(
      var__block2_sub_block_conv2_conv_output_0,
      { label: '/block2/sub_block/Sigmoid' }
    );
    
    const var__block2_sub_block_mul_output_0 = builder.mul(
      var__block2_sub_block_hardswish_output_0,
      var__block2_sub_block_sigmoid_output_0,
      { label: '/block2/sub_block/Mul' }
    );
    
    const var__block2_sub_block_conv3_conv_output_0 = builder.conv2d(
      var__block2_sub_block_mul_output_0, var_block2_sub_block_conv3_weight,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_block2_sub_block_conv3_bias,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: '/block2/sub_block/conv3/Conv'
      }
    );
    
    const var__block2_add_output_0 = builder.add(
      var__block1_add_output_0,
      var__block2_sub_block_conv3_conv_output_0,
      { label: '/block2/Add' }
    );
    
    const var__block3_conv1_conv_output_0 = builder.conv2d(
      var__block2_add_output_0, var_block3_conv1_weight,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_block3_conv1_bias,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: '/block3/conv1/Conv'
      }
    );
    
    const var__block3_hardswish_output_0 = builder.hardSwish(
      var__block3_conv1_conv_output_0,
      { label: '/block3/HardSwish' }
    );
    
    const var__block3_conv2_conv_output_0 = builder.conv2d(
      var__block3_hardswish_output_0, var_block3_conv2_weight,
      {
        strides: [1, 1],
        padding: [2, 2, 2, 2],
        dilations: [1, 1],
        groups: 96,
        bias: var_block3_conv2_bias,
        filterLayout: 'ihwo',
        inputLayout: 'nhwc',
        label: '/block3/conv2/Conv'
      }
    );
    
    const var__block3_sub_block_hardswish_output_0 = builder.hardSwish(
      var__block3_conv2_conv_output_0,
      { label: '/block3/sub_block/HardSwish' }
    );
    
    const var__block3_sub_block_reducemean_output_0 = builder.reduceMean(
      var__block3_sub_block_hardswish_output_0,
      { keepDimensions: true, axes: [1, 2], label: '/block3/sub_block/ReduceMean' }
    );
    
    const var__block3_sub_block_conv1_conv_output_0 = builder.conv2d(
      var__block3_sub_block_reducemean_output_0, var_block3_sub_block_conv1_weight,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_block3_sub_block_conv1_bias,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: '/block3/sub_block/conv1/Conv'
      }
    );
    
    const var__block3_sub_block_relu_output_0 = builder.relu(
      var__block3_sub_block_conv1_conv_output_0,
      { label: '/block3/sub_block/Relu' }
    );
    
    const var__block3_sub_block_conv2_conv_output_0 = builder.conv2d(
      var__block3_sub_block_relu_output_0, var_block3_sub_block_conv2_weight,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_block3_sub_block_conv2_bias,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: '/block3/sub_block/conv2/Conv'
      }
    );
    
    const var__block3_sub_block_sigmoid_output_0 = builder.sigmoid(
      var__block3_sub_block_conv2_conv_output_0,
      { label: '/block3/sub_block/Sigmoid' }
    );
    
    const var__block3_sub_block_mul_output_0 = builder.mul(
      var__block3_sub_block_hardswish_output_0,
      var__block3_sub_block_sigmoid_output_0,
      { label: '/block3/sub_block/Mul' }
    );
    
    const var__block3_sub_block_conv3_conv_output_0 = builder.conv2d(
      var__block3_sub_block_mul_output_0, var_block3_sub_block_conv3_weight,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_block3_sub_block_conv3_bias,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: '/block3/sub_block/conv3/Conv'
      }
    );
    
    const var__block3_add_output_0 = builder.add(
      var__block2_add_output_0,
      var__block3_sub_block_conv3_conv_output_0,
      { label: '/block3/Add' }
    );
    
    const var__block4_conv1_conv_output_0 = builder.conv2d(
      var__block3_add_output_0, var_block4_conv1_weight,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_block4_conv1_bias,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: '/block4/conv1/Conv'
      }
    );
    
    const var__block4_hardswish_output_0 = builder.hardSwish(
      var__block4_conv1_conv_output_0,
      { label: '/block4/HardSwish' }
    );
    
    const var__block4_conv2_conv_output_0 = builder.conv2d(
      var__block4_hardswish_output_0, var_block4_conv2_weight,
      {
        strides: [1, 1],
        padding: [2, 2, 2, 2],
        dilations: [1, 1],
        groups: 96,
        bias: var_block4_conv2_bias,
        filterLayout: 'ihwo',
        inputLayout: 'nhwc',
        label: '/block4/conv2/Conv'
      }
    );
    
    const var__block4_sub_block_hardswish_output_0 = builder.hardSwish(
      var__block4_conv2_conv_output_0,
      { label: '/block4/sub_block/HardSwish' }
    );
    
    const var__block4_sub_block_reducemean_output_0 = builder.reduceMean(
      var__block4_sub_block_hardswish_output_0,
      { keepDimensions: true, axes: [1, 2], label: '/block4/sub_block/ReduceMean' }
    );
    
    const var__block4_sub_block_conv1_conv_output_0 = builder.conv2d(
      var__block4_sub_block_reducemean_output_0, var_block4_sub_block_conv1_weight,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_block4_sub_block_conv1_bias,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: '/block4/sub_block/conv1/Conv'
      }
    );
    
    const var__block4_sub_block_relu_output_0 = builder.relu(
      var__block4_sub_block_conv1_conv_output_0,
      { label: '/block4/sub_block/Relu' }
    );
    
    const var__block4_sub_block_conv2_conv_output_0 = builder.conv2d(
      var__block4_sub_block_relu_output_0, var_block4_sub_block_conv2_weight,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_block4_sub_block_conv2_bias,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: '/block4/sub_block/conv2/Conv'
      }
    );
    
    const var__block4_sub_block_sigmoid_output_0 = builder.sigmoid(
      var__block4_sub_block_conv2_conv_output_0,
      { label: '/block4/sub_block/Sigmoid' }
    );
    
    const var__block4_sub_block_mul_output_0 = builder.mul(
      var__block4_sub_block_hardswish_output_0,
      var__block4_sub_block_sigmoid_output_0,
      { label: '/block4/sub_block/Mul' }
    );
    
    const var__block4_sub_block_conv3_conv_output_0 = builder.conv2d(
      var__block4_sub_block_mul_output_0, var_block4_sub_block_conv3_weight,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_block4_sub_block_conv3_bias,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: '/block4/sub_block/conv3/Conv'
      }
    );
    
    const var__block4_add_output_0 = builder.add(
      var__block3_add_output_0,
      var__block4_sub_block_conv3_conv_output_0,
      { label: '/block4/Add' }
    );
    
    const var__conv7_conv_output_0 = builder.conv2d(
      var__block4_add_output_0, var_conv7_weight,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_conv7_bias,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: '/conv7/Conv'
      }
    );
    
    const var__reducemean_1_output_0 = builder.reduceMean(
      var__block4_add_output_0,
      { keepDimensions: true, axes: [1, 2], label: '/ReduceMean_1' }
    );
    
    const var__relu_7_output_0 = builder.relu(
      var__conv7_conv_output_0,
      { label: '/Relu_7' }
    );
    
    const var__conv8_conv_output_0 = builder.conv2d(
      var__reducemean_1_output_0, var_conv8_weight,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_conv8_bias,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: '/conv8/Conv'
      }
    );
    
    const var__sigmoid_1_output_0 = builder.sigmoid(
      var__conv8_conv_output_0,
      { label: '/Sigmoid_1' }
    );
    
    const var__mul_1_output_0 = builder.mul(
      var__relu_7_output_0,
      var__sigmoid_1_output_0,
      { label: '/Mul_1' }
    );
    
    const var__final_block1_interp_resize_output_0 = builder.resample2d(
      var__mul_1_output_0,
      {
        mode: 'linear',
        sizes: [32, 32],
        axes: [1, 2],
        label: '/final_block1/interp/Resize'
      }
    );
    
    const var__final_block1_conv1_conv_output_0 = builder.conv2d(
      var__final_block1_interp_resize_output_0, var_final_block1_conv1_weight,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_final_block1_conv1_bias,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: '/final_block1/conv1/Conv'
      }
    );
    
    const var__final_block1_add_output_0 = builder.add(
      var__final_block1_conv1_conv_output_0,
      var__add_output_0,
      { label: '/final_block1/Add' }
    );
    
    const var__final_block1_reducemean_output_0 = builder.reduceMean(
      var__final_block1_add_output_0,
      { keepDimensions: true, axes: [1, 2], label: '/final_block1/ReduceMean' }
    );
    
    const var__final_block1_conv2_conv_output_0 = builder.conv2d(
      var__final_block1_reducemean_output_0, var_final_block1_conv2_weight,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_final_block1_conv2_bias,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: '/final_block1/conv2/Conv'
      }
    );
    
    const var__final_block1_relu_output_0 = builder.relu(
      var__final_block1_conv2_conv_output_0,
      { label: '/final_block1/Relu' }
    );
    
    const var__final_block1_conv3_conv_output_0 = builder.conv2d(
      var__final_block1_relu_output_0, var_final_block1_conv3_weight,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_final_block1_conv3_bias,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: '/final_block1/conv3/Conv'
      }
    );
    
    const var__final_block1_sigmoid_output_0 = builder.sigmoid(
      var__final_block1_conv3_conv_output_0,
      { label: '/final_block1/Sigmoid' }
    );
    
    const var__final_block1_mul_output_0 = builder.mul(
      var__add_output_0,
      var__final_block1_sigmoid_output_0,
      { label: '/final_block1/Mul' }
    );
    
    const var__final_block1_add_1_output_0 = builder.add(
      var__final_block1_conv1_conv_output_0,
      var__final_block1_mul_output_0,
      { label: '/final_block1/Add_1' }
    );
    
    const var__final_block1_conv4_conv_output_0 = builder.conv2d(
      var__final_block1_add_1_output_0, var_final_block1_conv4_weight,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_final_block1_conv4_bias,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: '/final_block1/conv4/Conv'
      }
    );
    
    const var__final_block1_relu_1_output_0 = builder.relu(
      var__final_block1_conv4_conv_output_0,
      { label: '/final_block1/Relu_1' }
    );
    
    const var__final_block1_conv5_conv_output_0 = builder.conv2d(
      var__final_block1_relu_1_output_0, var_final_block1_conv5_weight,
      {
        strides: [1, 1],
        padding: [1, 1, 1, 1],
        dilations: [1, 1],
        groups: 24,
        bias: var_final_block1_conv5_bias,
        filterLayout: 'ihwo',
        inputLayout: 'nhwc',
        label: '/final_block1/conv5/Conv'
      }
    );
    
    const var__final_block1_relu_2_output_0 = builder.relu(
      var__final_block1_conv5_conv_output_0,
      { label: '/final_block1/Relu_2' }
    );
    
    const var__final_block1_add_2_output_0 = builder.add(
      var__final_block1_relu_1_output_0,
      var__final_block1_relu_2_output_0,
      { label: '/final_block1/Add_2' }
    );
    
    const var__final_block2_interp_resize_output_0 = builder.resample2d(
      var__final_block1_add_2_output_0,
      {
        mode: 'linear',
        sizes: [64, 64],
        axes: [1, 2],
        label: '/final_block2/interp/Resize'
      }
    );
    
    const var__final_block2_conv1_conv_output_0 = builder.conv2d(
      var__final_block2_interp_resize_output_0, var_final_block2_conv1_weight,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_final_block2_conv1_bias,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: '/final_block2/conv1/Conv'
      }
    );
    
    const var__final_block2_add_output_0 = builder.add(
      var__final_block2_conv1_conv_output_0,
      var__conv3_3_conv_output_0,
      { label: '/final_block2/Add' }
    );
    
    const var__final_block2_reducemean_output_0 = builder.reduceMean(
      var__final_block2_add_output_0,
      { keepDimensions: true, axes: [1, 2], label: '/final_block2/ReduceMean' }
    );
    
    const var__final_block2_conv2_conv_output_0 = builder.conv2d(
      var__final_block2_reducemean_output_0, var_final_block2_conv2_weight,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_final_block2_conv2_bias,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: '/final_block2/conv2/Conv'
      }
    );
    
    const var__final_block2_relu_output_0 = builder.relu(
      var__final_block2_conv2_conv_output_0,
      { label: '/final_block2/Relu' }
    );
    
    const var__final_block2_conv3_conv_output_0 = builder.conv2d(
      var__final_block2_relu_output_0, var_final_block2_conv3_weight,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_final_block2_conv3_bias,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: '/final_block2/conv3/Conv'
      }
    );
    
    const var__final_block2_sigmoid_output_0 = builder.sigmoid(
      var__final_block2_conv3_conv_output_0,
      { label: '/final_block2/Sigmoid' }
    );
    
    const var__final_block2_mul_output_0 = builder.mul(
      var__conv3_3_conv_output_0,
      var__final_block2_sigmoid_output_0,
      { label: '/final_block2/Mul' }
    );
    
    const var__final_block2_add_1_output_0 = builder.add(
      var__final_block2_conv1_conv_output_0,
      var__final_block2_mul_output_0,
      { label: '/final_block2/Add_1' }
    );
    
    const var__final_block2_conv4_conv_output_0 = builder.conv2d(
      var__final_block2_add_1_output_0, var_final_block2_conv4_weight,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_final_block2_conv4_bias,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: '/final_block2/conv4/Conv'
      }
    );
    
    const var__final_block2_relu_1_output_0 = builder.relu(
      var__final_block2_conv4_conv_output_0,
      { label: '/final_block2/Relu_1' }
    );
    
    const var__final_block2_conv5_conv_output_0 = builder.conv2d(
      var__final_block2_relu_1_output_0, var_final_block2_conv5_weight,
      {
        strides: [1, 1],
        padding: [1, 1, 1, 1],
        dilations: [1, 1],
        groups: 16,
        bias: var_final_block2_conv5_bias,
        filterLayout: 'ihwo',
        inputLayout: 'nhwc',
        label: '/final_block2/conv5/Conv'
      }
    );
    
    const var__final_block2_relu_2_output_0 = builder.relu(
      var__final_block2_conv5_conv_output_0,
      { label: '/final_block2/Relu_2' }
    );
    
    const var__final_block2_add_2_output_0 = builder.add(
      var__final_block2_relu_1_output_0,
      var__final_block2_relu_2_output_0,
      { label: '/final_block2/Add_2' }
    );
    
    const var__final_block3_interp_resize_output_0 = builder.resample2d(
      var__final_block2_add_2_output_0,
      {
        mode: 'linear',
        sizes: [128, 128],
        axes: [1, 2],
        label: '/final_block3/interp/Resize'
      }
    );
    
    const var__final_block3_conv1_conv_output_0 = builder.conv2d(
      var__final_block3_interp_resize_output_0, var_final_block3_conv1_weight,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_final_block3_conv1_bias,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: '/final_block3/conv1/Conv'
      }
    );
    
    const var__final_block3_add_output_0 = builder.add(
      var__final_block3_conv1_conv_output_0,
      var__hardswish_output_0,
      { label: '/final_block3/Add' }
    );
    
    const var__final_block3_reducemean_output_0 = builder.reduceMean(
      var__final_block3_add_output_0,
      { keepDimensions: true, axes: [1, 2], label: '/final_block3/ReduceMean' }
    );
    
    const var__final_block3_conv2_conv_output_0 = builder.conv2d(
      var__final_block3_reducemean_output_0, var_final_block3_conv2_weight,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_final_block3_conv2_bias,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: '/final_block3/conv2/Conv'
      }
    );
    
    const var__final_block3_relu_output_0 = builder.relu(
      var__final_block3_conv2_conv_output_0,
      { label: '/final_block3/Relu' }
    );
    
    const var__final_block3_conv3_conv_output_0 = builder.conv2d(
      var__final_block3_relu_output_0, var_final_block3_conv3_weight,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_final_block3_conv3_bias,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: '/final_block3/conv3/Conv'
      }
    );
    
    const var__final_block3_sigmoid_output_0 = builder.sigmoid(
      var__final_block3_conv3_conv_output_0,
      { label: '/final_block3/Sigmoid' }
    );
    
    const var__final_block3_mul_output_0 = builder.mul(
      var__hardswish_output_0,
      var__final_block3_sigmoid_output_0,
      { label: '/final_block3/Mul' }
    );
    
    const var__final_block3_add_1_output_0 = builder.add(
      var__final_block3_conv1_conv_output_0,
      var__final_block3_mul_output_0,
      { label: '/final_block3/Add_1' }
    );
    
    const var__final_block3_conv4_conv_output_0 = builder.conv2d(
      var__final_block3_add_1_output_0, var_final_block3_conv4_weight,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_final_block3_conv4_bias,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: '/final_block3/conv4/Conv'
      }
    );
    
    const var__final_block3_relu_1_output_0 = builder.relu(
      var__final_block3_conv4_conv_output_0,
      { label: '/final_block3/Relu_1' }
    );
    
    const var__final_block3_conv5_conv_output_0 = builder.conv2d(
      var__final_block3_relu_1_output_0, var_final_block3_conv5_weight,
      {
        strides: [1, 1],
        padding: [1, 1, 1, 1],
        dilations: [1, 1],
        groups: 16,
        bias: var_final_block3_conv5_bias,
        filterLayout: 'ihwo',
        inputLayout: 'nhwc',
        label: '/final_block3/conv5/Conv'
      }
    );
    
    const var__final_block3_relu_2_output_0 = builder.relu(
      var__final_block3_conv5_conv_output_0,
      { label: '/final_block3/Relu_2' }
    );
    
    const var__final_block3_add_2_output_0 = builder.add(
      var__final_block3_relu_1_output_0,
      var__final_block3_relu_2_output_0,
      { label: '/final_block3/Add_2' }
    );
    
    const var__conv_transpose_convtranspose_output_0 = builder.convTranspose2d(
      var__final_block3_add_2_output_0, var_conv_transpose_weight,
      {
        strides: [2, 2],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        outputPadding: [0, 0],
        bias: var_conv_transpose_bias,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: '/conv_transpose/ConvTranspose'
      }
    );
    
    const alphas = builder.sigmoid(
      var__conv_transpose_convtranspose_output_0,
      { label: '/Sigmoid_2' }
    );

    // Build graph with all outputs
    
    const alphas_nchw = builder.transpose(alphas, { permutation: [0, 3, 1, 2] });
    this.graph_ = await builder.build({ 'alphas': alphas_nchw });

    // Create output tensors
    
    this.outputTensors_['alphas'] = await this.context_.createTensor(
      { dataType: 'float32', shape: [1,1,256,256], readable: true }
    );
    
  }

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
  }
}