// WebNN Code Generator (NHWC)

export class SelfieSegmenterLandscape19Nhwc {

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
    
    const input_nchw = builder.input('input', { dataType: 'float32', shape: [1,144,256,3] });
    const input = builder.transpose(input_nchw, { permutation: [0, 2, 3, 1] });
    this.inputTensors_['input'] = await this.context_.createTensor(
      { dataType: 'float32', shape: [1,144,256,3], writable: true }
    );

    // Create graph constant operands
    
    // index.ts line 145
    const var_conv2d_filter_0 = builder.constant(
      { dataType: 'float32', shape: [16,3,3,3] },
      new Float32Array(weights_array_buffer, 0, 1728 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    // index.ts line 145
    const var_conv2d_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [16] },
      new Float32Array(weights_array_buffer, 1728, 64 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    // index.ts line 145
    const const_fold_opt__408 = builder.constant(
      { dataType: 'float32', shape: [1,1,1,1] },
      new Float32Array(weights_array_buffer, 329504, 4 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    // index.ts line 162
    const var_292 = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0])
    );

    // index.ts line 162
    const var_293 = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([6])
    );

    // index.ts line 145
    const var_mul_3_y_0 = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array(weights_array_buffer, 329528, 4 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    // index.ts line 145
    const var_conv2d_2_filter_0 = builder.constant(
      { dataType: 'float32', shape: [16,1,1,16] },
      new Float32Array(weights_array_buffer, 1824, 1024 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    // index.ts line 145
    const var_conv2d_1_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [16] },
      new Float32Array(weights_array_buffer, 2848, 64 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    // index.ts line 145
    const const_fold_opt__377 = builder.constant(
      { dataType: 'float32', shape: [16,3,3,1] },
      new Float32Array(weights_array_buffer, 2912, 576 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    // index.ts line 145
    const var_depthwise_conv2d_y_0 = builder.constant(
      { dataType: 'float32', shape: [16] },
      new Float32Array(weights_array_buffer, 3488, 64 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    // index.ts line 145
    const var_conv2d_3_filter_0 = builder.constant(
      { dataType: 'float32', shape: [8,1,1,16] },
      new Float32Array(weights_array_buffer, 3552, 512 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    // index.ts line 145
    const var_conv2d_2_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [8] },
      new Float32Array(weights_array_buffer, 4064, 32 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    // index.ts line 145
    const var_conv2d_4_filter_0 = builder.constant(
      { dataType: 'float32', shape: [16,1,1,8] },
      new Float32Array(weights_array_buffer, 4096, 512 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    // index.ts line 145
    const var_conv2d_3_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [16] },
      new Float32Array(weights_array_buffer, 4608, 64 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    // index.ts line 145
    const var_conv2d_5_filter_0 = builder.constant(
      { dataType: 'float32', shape: [16,1,1,16] },
      new Float32Array(weights_array_buffer, 4672, 1024 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    // index.ts line 145
    const var_conv2d_4_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [16] },
      new Float32Array(weights_array_buffer, 5696, 64 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    // index.ts line 145
    const var_conv2d_6_filter_0 = builder.constant(
      { dataType: 'float32', shape: [72,1,1,16] },
      new Float32Array(weights_array_buffer, 5760, 4608 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    // index.ts line 145
    const var_conv2d_5_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [72] },
      new Float32Array(weights_array_buffer, 10368, 288 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    // index.ts line 145
    const const_fold_opt__387 = builder.constant(
      { dataType: 'float32', shape: [72,3,3,1] },
      new Float32Array(weights_array_buffer, 10656, 2592 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    // index.ts line 145
    const var_depthwise_conv2d_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [72] },
      new Float32Array(weights_array_buffer, 13248, 288 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    // index.ts line 145
    const var_conv2d_7_filter_0 = builder.constant(
      { dataType: 'float32', shape: [24,1,1,72] },
      new Float32Array(weights_array_buffer, 13536, 6912 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    // index.ts line 145
    const var_conv2d_6_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [24] },
      new Float32Array(weights_array_buffer, 20448, 96 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    // index.ts line 145
    const var_conv2d_8_filter_0 = builder.constant(
      { dataType: 'float32', shape: [88,1,1,24] },
      new Float32Array(weights_array_buffer, 20544, 8448 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    // index.ts line 145
    const var_conv2d_7_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [88] },
      new Float32Array(weights_array_buffer, 28992, 352 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    // index.ts line 145
    const const_fold_opt__385 = builder.constant(
      { dataType: 'float32', shape: [88,3,3,1] },
      new Float32Array(weights_array_buffer, 29344, 3168 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    // index.ts line 145
    const var_depthwise_conv2d_2_y_0 = builder.constant(
      { dataType: 'float32', shape: [88] },
      new Float32Array(weights_array_buffer, 32512, 352 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    // index.ts line 145
    const var_conv2d_9_filter_0 = builder.constant(
      { dataType: 'float32', shape: [24,1,1,88] },
      new Float32Array(weights_array_buffer, 32864, 8448 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    // index.ts line 145
    const var_conv2d_8_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [24] },
      new Float32Array(weights_array_buffer, 41312, 96 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    // index.ts line 145
    const var_conv2d_10_filter_0 = builder.constant(
      { dataType: 'float32', shape: [96,1,1,24] },
      new Float32Array(weights_array_buffer, 41408, 9216 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    // index.ts line 145
    const var_conv2d_9_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [96] },
      new Float32Array(weights_array_buffer, 50624, 384 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    // index.ts line 162
    const var_294 = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0])
    );

    // index.ts line 162
    const var_295 = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([6])
    );

    // index.ts line 145
    const const_fold_opt__383 = builder.constant(
      { dataType: 'float32', shape: [96,5,5,1] },
      new Float32Array(weights_array_buffer, 51040, 9600 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    // index.ts line 145
    const var_depthwise_conv2d_3_y_0 = builder.constant(
      { dataType: 'float32', shape: [96] },
      new Float32Array(weights_array_buffer, 60640, 384 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    // index.ts line 162
    const var_296 = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0])
    );

    // index.ts line 162
    const var_297 = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([6])
    );

    // index.ts line 145
    const var_conv2d_11_filter_0 = builder.constant(
      { dataType: 'float32', shape: [24,1,1,96] },
      new Float32Array(weights_array_buffer, 61056, 9216 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    // index.ts line 145
    const var_conv2d_10_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [24] },
      new Float32Array(weights_array_buffer, 70272, 96 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    // index.ts line 145
    const var_conv2d_12_filter_0 = builder.constant(
      { dataType: 'float32', shape: [96,1,1,24] },
      new Float32Array(weights_array_buffer, 70368, 9216 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    // index.ts line 145
    const var_conv2d_11_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [96] },
      new Float32Array(weights_array_buffer, 79584, 384 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    // index.ts line 145
    const var_conv2d_13_filter_0 = builder.constant(
      { dataType: 'float32', shape: [32,1,1,96] },
      new Float32Array(weights_array_buffer, 79968, 12288 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    // index.ts line 145
    const var_conv2d_12_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [32] },
      new Float32Array(weights_array_buffer, 92256, 128 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    // index.ts line 145
    const var_conv2d_14_filter_0 = builder.constant(
      { dataType: 'float32', shape: [128,1,1,32] },
      new Float32Array(weights_array_buffer, 92384, 16384 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    // index.ts line 145
    const var_conv2d_13_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [128] },
      new Float32Array(weights_array_buffer, 108768, 512 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    // index.ts line 162
    const var_298 = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0])
    );

    // index.ts line 162
    const var_299 = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([6])
    );

    // index.ts line 145
    const const_fold_opt__381 = builder.constant(
      { dataType: 'float32', shape: [128,5,5,1] },
      new Float32Array(weights_array_buffer, 109312, 12800 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    // index.ts line 145
    const var_depthwise_conv2d_4_y_0 = builder.constant(
      { dataType: 'float32', shape: [128] },
      new Float32Array(weights_array_buffer, 122112, 512 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    // index.ts line 162
    const var_300 = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0])
    );

    // index.ts line 162
    const var_301 = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([6])
    );

    // index.ts line 145
    const var_conv2d_15_filter_0 = builder.constant(
      { dataType: 'float32', shape: [32,1,1,128] },
      new Float32Array(weights_array_buffer, 122656, 16384 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    // index.ts line 145
    const var_conv2d_14_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [32] },
      new Float32Array(weights_array_buffer, 139040, 128 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    // index.ts line 145
    const var_conv2d_16_filter_0 = builder.constant(
      { dataType: 'float32', shape: [128,1,1,32] },
      new Float32Array(weights_array_buffer, 139168, 16384 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    // index.ts line 145
    const var_conv2d_15_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [128] },
      new Float32Array(weights_array_buffer, 155552, 512 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    // index.ts line 145
    const var_conv2d_17_filter_0 = builder.constant(
      { dataType: 'float32', shape: [32,1,1,128] },
      new Float32Array(weights_array_buffer, 156064, 16384 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    // index.ts line 145
    const var_conv2d_16_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [32] },
      new Float32Array(weights_array_buffer, 172448, 128 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    // index.ts line 145
    const var_conv2d_18_filter_0 = builder.constant(
      { dataType: 'float32', shape: [128,1,1,32] },
      new Float32Array(weights_array_buffer, 172576, 16384 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    // index.ts line 145
    const var_conv2d_17_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [128] },
      new Float32Array(weights_array_buffer, 188960, 512 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    // index.ts line 162
    const var_302 = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0])
    );

    // index.ts line 162
    const var_303 = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([6])
    );

    // index.ts line 145
    const const_fold_opt__379 = builder.constant(
      { dataType: 'float32', shape: [128,5,5,1] },
      new Float32Array(weights_array_buffer, 189504, 12800 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    // index.ts line 145
    const var_depthwise_conv2d_5_y_0 = builder.constant(
      { dataType: 'float32', shape: [128] },
      new Float32Array(weights_array_buffer, 202304, 512 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    // index.ts line 162
    const var_304 = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0])
    );

    // index.ts line 162
    const var_305 = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([6])
    );

    // index.ts line 145
    const var_conv2d_19_filter_0 = builder.constant(
      { dataType: 'float32', shape: [32,1,1,128] },
      new Float32Array(weights_array_buffer, 202848, 16384 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    // index.ts line 145
    const var_conv2d_18_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [32] },
      new Float32Array(weights_array_buffer, 219232, 128 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    // index.ts line 145
    const var_conv2d_20_filter_0 = builder.constant(
      { dataType: 'float32', shape: [128,1,1,32] },
      new Float32Array(weights_array_buffer, 219360, 16384 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    // index.ts line 145
    const var_conv2d_19_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [128] },
      new Float32Array(weights_array_buffer, 235744, 512 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    // index.ts line 145
    const var_conv2d_21_filter_0 = builder.constant(
      { dataType: 'float32', shape: [32,1,1,128] },
      new Float32Array(weights_array_buffer, 236256, 16384 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    // index.ts line 145
    const var_conv2d_20_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [32] },
      new Float32Array(weights_array_buffer, 252640, 128 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    // index.ts line 145
    const var_conv2d_22_filter_0 = builder.constant(
      { dataType: 'float32', shape: [96,1,1,32] },
      new Float32Array(weights_array_buffer, 252768, 12288 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    // index.ts line 145
    const var_conv2d_21_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [96] },
      new Float32Array(weights_array_buffer, 265056, 384 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    // index.ts line 162
    const var_306 = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0])
    );

    // index.ts line 162
    const var_307 = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([6])
    );

    // index.ts line 145
    const const_fold_opt__395 = builder.constant(
      { dataType: 'float32', shape: [96,5,5,1] },
      new Float32Array(weights_array_buffer, 265472, 9600 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    // index.ts line 145
    const var_depthwise_conv2d_6_y_0 = builder.constant(
      { dataType: 'float32', shape: [96] },
      new Float32Array(weights_array_buffer, 275072, 384 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    // index.ts line 162
    const var_308 = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0])
    );

    // index.ts line 162
    const var_309 = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([6])
    );

    // index.ts line 145
    const var_conv2d_23_filter_0 = builder.constant(
      { dataType: 'float32', shape: [24,1,1,96] },
      new Float32Array(weights_array_buffer, 275488, 9216 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    // index.ts line 145
    const var_conv2d_22_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [24] },
      new Float32Array(weights_array_buffer, 284704, 96 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    // index.ts line 145
    const var_conv2d_24_filter_0 = builder.constant(
      { dataType: 'float32', shape: [96,1,1,24] },
      new Float32Array(weights_array_buffer, 284800, 9216 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    // index.ts line 145
    const var_conv2d_23_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [96] },
      new Float32Array(weights_array_buffer, 294016, 384 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    // index.ts line 145
    const var_conv2d_25_filter_0 = builder.constant(
      { dataType: 'float32', shape: [32,1,1,96] },
      new Float32Array(weights_array_buffer, 294400, 12288 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    // index.ts line 145
    const var_conv2d_24_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [32] },
      new Float32Array(weights_array_buffer, 306688, 128 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    // index.ts line 145
    const var_conv2d_26_filter_0 = builder.constant(
      { dataType: 'float32', shape: [96,1,1,32] },
      new Float32Array(weights_array_buffer, 306816, 12288 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    // index.ts line 145
    const var_conv2d_25_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [96] },
      new Float32Array(weights_array_buffer, 319104, 384 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    // index.ts line 162
    const var_310 = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0])
    );

    // index.ts line 162
    const var_311 = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([6])
    );

    // index.ts line 145
    const const_fold_opt__407 = builder.constant(
      { dataType: 'float32', shape: [96,5,5,1] },
      new Float32Array(weights_array_buffer, 319520, 9600 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    // index.ts line 145
    const var_depthwise_conv2d_7_y_0 = builder.constant(
      { dataType: 'float32', shape: [96] },
      new Float32Array(weights_array_buffer, 329120, 384 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    // index.ts line 162
    const var_312 = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0])
    );

    // index.ts line 162
    const var_313 = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([6])
    );

    // index.ts line 145
    const var_conv2d_27_filter_0 = builder.constant(
      { dataType: 'float32', shape: [24,1,1,96] },
      new Float32Array(weights_array_buffer, 329536, 9216 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    // index.ts line 145
    const var_conv2d_26_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [24] },
      new Float32Array(weights_array_buffer, 338752, 96 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    // index.ts line 145
    const var_conv2d_28_filter_0 = builder.constant(
      { dataType: 'float32', shape: [96,1,1,24] },
      new Float32Array(weights_array_buffer, 338848, 9216 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    // index.ts line 145
    const var_conv2d_27_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [96] },
      new Float32Array(weights_array_buffer, 348064, 384 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    // index.ts line 145
    const var_conv2d_29_filter_0 = builder.constant(
      { dataType: 'float32', shape: [32,1,1,96] },
      new Float32Array(weights_array_buffer, 348448, 12288 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    // index.ts line 145
    const var_conv2d_28_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [32] },
      new Float32Array(weights_array_buffer, 360736, 128 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    // index.ts line 145
    const var_conv2d_31_filter_0 = builder.constant(
      { dataType: 'float32', shape: [128,1,1,32] },
      new Float32Array(weights_array_buffer, 360864, 16384 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    // index.ts line 145
    const var_conv2d_30_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [128] },
      new Float32Array(weights_array_buffer, 377248, 512 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    // index.ts line 145
    const var_conv2d_30_filter_0 = builder.constant(
      { dataType: 'float32', shape: [128,1,1,32] },
      new Float32Array(weights_array_buffer, 377760, 16384 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    // index.ts line 145
    const var_conv2d_29_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [128] },
      new Float32Array(weights_array_buffer, 394144, 512 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    // index.ts line 162
    const var_314 = builder.constant(
      { dataType: 'float32', shape: [8] },
      new Float32Array([0, 0, 0, 0, 1, 1, 1, 1])
    );

    // index.ts line 162
    const scales__125 = builder.constant(
      { dataType: 'float32', shape: [4] },
      new Float32Array([1, 1, 2, 2])
    );

    // index.ts line 145
    const var_conv2d_32_filter_0 = builder.constant(
      { dataType: 'float32', shape: [24,1,1,128] },
      new Float32Array(weights_array_buffer, 394704, 12288 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    // index.ts line 145
    const var_conv2d_31_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [24] },
      new Float32Array(weights_array_buffer, 406992, 96 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    // index.ts line 145
    const var_conv2d_33_filter_0 = builder.constant(
      { dataType: 'float32', shape: [24,1,1,24] },
      new Float32Array(weights_array_buffer, 407088, 2304 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    // index.ts line 145
    const var_conv2d_32_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [24] },
      new Float32Array(weights_array_buffer, 409392, 96 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    // index.ts line 145
    const var_conv2d_34_filter_0 = builder.constant(
      { dataType: 'float32', shape: [24,1,1,24] },
      new Float32Array(weights_array_buffer, 409488, 2304 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    // index.ts line 145
    const var_conv2d_33_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [24] },
      new Float32Array(weights_array_buffer, 411792, 96 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    // index.ts line 145
    const var_conv2d_35_filter_0 = builder.constant(
      { dataType: 'float32', shape: [24,1,1,24] },
      new Float32Array(weights_array_buffer, 411888, 2304 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    // index.ts line 145
    const var_conv2d_34_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [24] },
      new Float32Array(weights_array_buffer, 414192, 96 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    // index.ts line 145
    const const_fold_opt__394 = builder.constant(
      { dataType: 'float32', shape: [24,3,3,1] },
      new Float32Array(weights_array_buffer, 414288, 864 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    // index.ts line 145
    const var_depthwise_conv2d_8_y_0 = builder.constant(
      { dataType: 'float32', shape: [24] },
      new Float32Array(weights_array_buffer, 415152, 96 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    // index.ts line 162
    const var_315 = builder.constant(
      { dataType: 'float32', shape: [8] },
      new Float32Array([0, 0, 0, 0, 1, 1, 1, 1])
    );

    // index.ts line 145
    const var_conv2d_36_filter_0 = builder.constant(
      { dataType: 'float32', shape: [16,1,1,24] },
      new Float32Array(weights_array_buffer, 415296, 1536 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    // index.ts line 145
    const var_conv2d_35_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [16] },
      new Float32Array(weights_array_buffer, 416832, 64 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    // index.ts line 145
    const var_conv2d_37_filter_0 = builder.constant(
      { dataType: 'float32', shape: [16,1,1,16] },
      new Float32Array(weights_array_buffer, 416896, 1024 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    // index.ts line 145
    const var_conv2d_36_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [16] },
      new Float32Array(weights_array_buffer, 417920, 64 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    // index.ts line 145
    const var_conv2d_38_filter_0 = builder.constant(
      { dataType: 'float32', shape: [16,1,1,16] },
      new Float32Array(weights_array_buffer, 417984, 1024 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    // index.ts line 145
    const var_conv2d_37_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [16] },
      new Float32Array(weights_array_buffer, 419008, 64 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    // index.ts line 145
    const var_conv2d_39_filter_0 = builder.constant(
      { dataType: 'float32', shape: [16,1,1,16] },
      new Float32Array(weights_array_buffer, 419072, 1024 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    // index.ts line 145
    const var_conv2d_38_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [16] },
      new Float32Array(weights_array_buffer, 420096, 64 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    // index.ts line 145
    const const_fold_opt__391 = builder.constant(
      { dataType: 'float32', shape: [16,3,3,1] },
      new Float32Array(weights_array_buffer, 420160, 576 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    // index.ts line 145
    const var_depthwise_conv2d_9_y_0 = builder.constant(
      { dataType: 'float32', shape: [16] },
      new Float32Array(weights_array_buffer, 420736, 64 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    // index.ts line 162
    const var_316 = builder.constant(
      { dataType: 'float32', shape: [8] },
      new Float32Array([0, 0, 0, 0, 1, 1, 1, 1])
    );

    // index.ts line 145
    const var_conv2d_40_filter_0 = builder.constant(
      { dataType: 'float32', shape: [16,1,1,16] },
      new Float32Array(weights_array_buffer, 420848, 1024 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    // index.ts line 145
    const var_conv2d_39_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [16] },
      new Float32Array(weights_array_buffer, 421872, 64 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    // index.ts line 145
    const var_conv2d_41_filter_0 = builder.constant(
      { dataType: 'float32', shape: [16,1,1,16] },
      new Float32Array(weights_array_buffer, 421936, 1024 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    // index.ts line 145
    const var_conv2d_40_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [16] },
      new Float32Array(weights_array_buffer, 422960, 64 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    // index.ts line 145
    const var_conv2d_42_filter_0 = builder.constant(
      { dataType: 'float32', shape: [16,1,1,16] },
      new Float32Array(weights_array_buffer, 423024, 1024 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    // index.ts line 145
    const var_conv2d_41_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [16] },
      new Float32Array(weights_array_buffer, 424048, 64 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    // index.ts line 145
    const var_conv2d_43_filter_0 = builder.constant(
      { dataType: 'float32', shape: [16,1,1,16] },
      new Float32Array(weights_array_buffer, 424112, 1024 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    // index.ts line 145
    const var_conv2d_42_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [16] },
      new Float32Array(weights_array_buffer, 425136, 64 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    // index.ts line 145
    const const_fold_opt__389 = builder.constant(
      { dataType: 'float32', shape: [16,3,3,1] },
      new Float32Array(weights_array_buffer, 425200, 576 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    // index.ts line 145
    const var_depthwise_conv2d_10_y_0 = builder.constant(
      { dataType: 'float32', shape: [16] },
      new Float32Array(weights_array_buffer, 425776, 64 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    // index.ts line 145
    const var_conv2d_transpose_filter_0 = builder.constant(
      { dataType: 'float32', shape: [16,2,2,1] },
      new Float32Array(weights_array_buffer, 425840, 256 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    // index.ts line 145
    const const_fold_opt__402 = builder.constant(
      { dataType: 'float32', shape: [1,1,1,1] },
      new Float32Array(weights_array_buffer, 426096, 4 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    // index.ts line 145
    const new_shape__369 = builder.constant(
      { dataType: 'int64', shape: [4] },
      new BigInt64Array(weights_array_buffer, 426104, 32 / BigInt64Array.BYTES_PER_ELEMENT)
    );
    

    // Create graph operators
        
    const var_conv2d__6_0 = builder.transpose(
      input,
      { permutation: [0, 3, 1, 2] }
    );
    
    const var_conv__158_0 = builder.conv2d(
      var_conv2d__6_0, var_conv2d_filter_0,
      {
        strides: [2, 2],
        padding: [0, 1, 0, 1],
        dilations: [1, 1],
        groups: 1,
        bias: var_conv2d_1_y_0,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: 'Conv__158'
      }
    );
  
    
    const var_add_0 = builder.add(
      var_conv__158_0,
      const_fold_opt__408
    );
    
    const var_relu6_0 = builder.clamp(
      var_add_0,
      {
        minValue: 0,
        maxValue: 6
      }
    );
    
    const var_mul_0 = builder.mul(
      var_mul_3_y_0,
      var_relu6_0
    );
    
    const var_mul_1_0 = builder.mul(
      var_conv__158_0,
      var_mul_0
    );
    
    const var_conv__161_0 = builder.conv2d(
      var_mul_1_0, var_conv2d_2_filter_0,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_conv2d_1_1_y_0,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: 'Conv__161'
      }
    );
  
    
    const var_re_lu_0 = builder.relu(
      var_conv__161_0,
      { label: 're_lu' }
    );
  
    
    const var_conv__162_0 = builder.conv2d(
      var_re_lu_0, const_fold_opt__377,
      {
        strides: [2, 2],
        padding: [0, 1, 0, 1],
        dilations: [1, 1],
        groups: 16,
        bias: var_depthwise_conv2d_y_0,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: 'Conv__162'
      }
    );
  
    
    const var_re_lu_1_0 = builder.relu(
      var_conv__162_0,
      { label: 're_lu_1' }
    );
  
    
    const var_global_average_pooling2d_0 = builder.averagePool2d(
      var_re_lu_1_0,
      { layout: 'nhwc' }
    );
    
    const var_conv__165_0 = builder.conv2d(
      var_global_average_pooling2d_0, var_conv2d_3_filter_0,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_conv2d_2_1_y_0,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: 'Conv__165'
      }
    );
  
    
    const var_re_lu_2_0 = builder.relu(
      var_conv__165_0,
      { label: 're_lu_2' }
    );
  
    
    const var_conv__166_0 = builder.conv2d(
      var_re_lu_2_0, var_conv2d_4_filter_0,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_conv2d_3_1_y_0,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: 'Conv__166'
      }
    );
  
    
    const var_activation_0 = builder.sigmoid(
      var_conv__166_0,
      { label: 'activation' }
    );
  
    
    const var_multiply_0 = builder.mul(
      var_re_lu_1_0,
      var_activation_0
    );
    
    const var_conv__167_0 = builder.conv2d(
      var_multiply_0, var_conv2d_5_filter_0,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_conv2d_4_1_y_0,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: 'Conv__167'
      }
    );
  
    
    const var_conv__171_0 = builder.conv2d(
      var_conv__167_0, var_conv2d_6_filter_0,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_conv2d_5_1_y_0,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: 'Conv__171'
      }
    );
  
    
    const var_re_lu_3_0 = builder.relu(
      var_conv__171_0,
      { label: 're_lu_3' }
    );
  
    
    const var_conv__172_0 = builder.conv2d(
      var_re_lu_3_0, const_fold_opt__387,
      {
        strides: [2, 2],
        padding: [0, 1, 0, 1],
        dilations: [1, 1],
        groups: 72,
        bias: var_depthwise_conv2d_1_y_0,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: 'Conv__172'
      }
    );
  
    
    const var_re_lu_4_0 = builder.relu(
      var_conv__172_0,
      { label: 're_lu_4' }
    );
  
    
    const var_conv__173_0 = builder.conv2d(
      var_re_lu_4_0, var_conv2d_7_filter_0,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_conv2d_6_1_y_0,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: 'Conv__173'
      }
    );
  
    
    const var_conv__176_0 = builder.conv2d(
      var_conv__173_0, var_conv2d_8_filter_0,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_conv2d_7_1_y_0,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: 'Conv__176'
      }
    );
  
    
    const var_re_lu_5_0 = builder.relu(
      var_conv__176_0,
      { label: 're_lu_5' }
    );
  
    
    const var_conv__177_0 = builder.conv2d(
      var_re_lu_5_0, const_fold_opt__385,
      {
        strides: [1, 1],
        padding: [1, 1, 1, 1],
        dilations: [1, 1],
        groups: 88,
        bias: var_depthwise_conv2d_2_y_0,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: 'Conv__177'
      }
    );
  
    
    const var_re_lu_6_0 = builder.relu(
      var_conv__177_0,
      { label: 're_lu_6' }
    );
  
    
    const var_conv__178_0 = builder.conv2d(
      var_re_lu_6_0, var_conv2d_9_filter_0,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_conv2d_8_1_y_0,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: 'Conv__178'
      }
    );
  
    
    const var_add__xeno_compat__1_0 = builder.add(
      var_conv__178_0,
      var_conv__173_0
    );
    
    const var_conv__183_0 = builder.conv2d(
      var_add__xeno_compat__1_0, var_conv2d_10_filter_0,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_conv2d_9_1_y_0,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: 'Conv__183'
      }
    );
  
    
    const var_add_1_0 = builder.add(
      var_conv__183_0,
      const_fold_opt__408
    );
    
    const var_relu6_1_0 = builder.clamp(
      var_add_1_0,
      {
        minValue: 0,
        maxValue: 6
      }
    );
    
    const var_mul_2_0 = builder.mul(
      var_mul_3_y_0,
      var_relu6_1_0
    );
    
    const var_mul_3_0 = builder.mul(
      var_conv__183_0,
      var_mul_2_0
    );
    
    const var_conv__186_0 = builder.conv2d(
      var_mul_3_0, const_fold_opt__383,
      {
        strides: [2, 2],
        padding: [1, 2, 1, 2],
        dilations: [1, 1],
        groups: 96,
        bias: var_depthwise_conv2d_3_y_0,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: 'Conv__186'
      }
    );
  
    
    const var_add_2_0 = builder.add(
      var_conv__186_0,
      const_fold_opt__408
    );
    
    const var_relu6_2_0 = builder.clamp(
      var_add_2_0,
      {
        minValue: 0,
        maxValue: 6
      }
    );
    
    const var_mul_4_0 = builder.mul(
      var_mul_3_y_0,
      var_relu6_2_0
    );
    
    const var_mul_5_0 = builder.mul(
      var_conv__186_0,
      var_mul_4_0
    );
    
    const var_global_average_pooling2d_1_0 = builder.averagePool2d(
      var_mul_5_0,
      { layout: 'nhwc' }
    );
    
    const var_conv__189_0 = builder.conv2d(
      var_global_average_pooling2d_1_0, var_conv2d_11_filter_0,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_conv2d_10_1_y_0,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: 'Conv__189'
      }
    );
  
    
    const var_re_lu_7_0 = builder.relu(
      var_conv__189_0,
      { label: 're_lu_7' }
    );
  
    
    const var_conv__190_0 = builder.conv2d(
      var_re_lu_7_0, var_conv2d_12_filter_0,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_conv2d_11_1_y_0,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: 'Conv__190'
      }
    );
  
    
    const var_activation_1_0 = builder.sigmoid(
      var_conv__190_0,
      { label: 'activation_1' }
    );
  
    
    const var_multiply_1_0 = builder.mul(
      var_mul_5_0,
      var_activation_1_0
    );
    
    const var_conv__191_0 = builder.conv2d(
      var_multiply_1_0, var_conv2d_13_filter_0,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_conv2d_12_1_y_0,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: 'Conv__191'
      }
    );
  
    
    const var_conv__194_0 = builder.conv2d(
      var_conv__191_0, var_conv2d_14_filter_0,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_conv2d_13_1_y_0,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: 'Conv__194'
      }
    );
  
    
    const var_add_3_0 = builder.add(
      var_conv__194_0,
      const_fold_opt__408
    );
    
    const var_relu6_3_0 = builder.clamp(
      var_add_3_0,
      {
        minValue: 0,
        maxValue: 6
      }
    );
    
    const var_mul_6_0 = builder.mul(
      var_mul_3_y_0,
      var_relu6_3_0
    );
    
    const var_mul_7_0 = builder.mul(
      var_conv__194_0,
      var_mul_6_0
    );
    
    const var_conv__197_0 = builder.conv2d(
      var_mul_7_0, const_fold_opt__381,
      {
        strides: [1, 1],
        padding: [2, 2, 2, 2],
        dilations: [1, 1],
        groups: 128,
        bias: var_depthwise_conv2d_4_y_0,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: 'Conv__197'
      }
    );
  
    
    const var_add_4_0 = builder.add(
      var_conv__197_0,
      const_fold_opt__408
    );
    
    const var_relu6_4_0 = builder.clamp(
      var_add_4_0,
      {
        minValue: 0,
        maxValue: 6
      }
    );
    
    const var_mul_8_0 = builder.mul(
      var_mul_3_y_0,
      var_relu6_4_0
    );
    
    const var_mul_9_0 = builder.mul(
      var_conv__197_0,
      var_mul_8_0
    );
    
    const var_global_average_pooling2d_2_0 = builder.averagePool2d(
      var_mul_9_0,
      { layout: 'nhwc' }
    );
    
    const var_conv__200_0 = builder.conv2d(
      var_global_average_pooling2d_2_0, var_conv2d_15_filter_0,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_conv2d_14_1_y_0,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: 'Conv__200'
      }
    );
  
    
    const var_re_lu_8_0 = builder.relu(
      var_conv__200_0,
      { label: 're_lu_8' }
    );
  
    
    const var_conv__201_0 = builder.conv2d(
      var_re_lu_8_0, var_conv2d_16_filter_0,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_conv2d_15_1_y_0,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: 'Conv__201'
      }
    );
  
    
    const var_activation_2_0 = builder.sigmoid(
      var_conv__201_0,
      { label: 'activation_2' }
    );
  
    
    const var_multiply_2_0 = builder.mul(
      var_mul_9_0,
      var_activation_2_0
    );
    
    const var_conv__202_0 = builder.conv2d(
      var_multiply_2_0, var_conv2d_17_filter_0,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_conv2d_16_1_y_0,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: 'Conv__202'
      }
    );
  
    
    const var_add_1__xeno_compat__1_0 = builder.add(
      var_conv__202_0,
      var_conv__191_0
    );
    
    const var_conv__205_0 = builder.conv2d(
      var_add_1__xeno_compat__1_0, var_conv2d_18_filter_0,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_conv2d_17_1_y_0,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: 'Conv__205'
      }
    );
  
    
    const var_add_5_0 = builder.add(
      var_conv__205_0,
      const_fold_opt__408
    );
    
    const var_relu6_5_0 = builder.clamp(
      var_add_5_0,
      {
        minValue: 0,
        maxValue: 6
      }
    );
    
    const var_mul_10_0 = builder.mul(
      var_mul_3_y_0,
      var_relu6_5_0
    );
    
    const var_mul_11_0 = builder.mul(
      var_conv__205_0,
      var_mul_10_0
    );
    
    const var_conv__208_0 = builder.conv2d(
      var_mul_11_0, const_fold_opt__379,
      {
        strides: [1, 1],
        padding: [2, 2, 2, 2],
        dilations: [1, 1],
        groups: 128,
        bias: var_depthwise_conv2d_5_y_0,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: 'Conv__208'
      }
    );
  
    
    const var_add_6_0 = builder.add(
      var_conv__208_0,
      const_fold_opt__408
    );
    
    const var_relu6_6_0 = builder.clamp(
      var_add_6_0,
      {
        minValue: 0,
        maxValue: 6
      }
    );
    
    const var_mul_12_0 = builder.mul(
      var_mul_3_y_0,
      var_relu6_6_0
    );
    
    const var_mul_13_0 = builder.mul(
      var_conv__208_0,
      var_mul_12_0
    );
    
    const var_global_average_pooling2d_3_0 = builder.averagePool2d(
      var_mul_13_0,
      { layout: 'nhwc' }
    );
    
    const var_conv__211_0 = builder.conv2d(
      var_global_average_pooling2d_3_0, var_conv2d_19_filter_0,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_conv2d_18_1_y_0,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: 'Conv__211'
      }
    );
  
    
    const var_re_lu_9_0 = builder.relu(
      var_conv__211_0,
      { label: 're_lu_9' }
    );
  
    
    const var_conv__212_0 = builder.conv2d(
      var_re_lu_9_0, var_conv2d_20_filter_0,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_conv2d_19_1_y_0,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: 'Conv__212'
      }
    );
  
    
    const var_activation_3_0 = builder.sigmoid(
      var_conv__212_0,
      { label: 'activation_3' }
    );
  
    
    const var_multiply_3_0 = builder.mul(
      var_mul_13_0,
      var_activation_3_0
    );
    
    const var_conv__213_0 = builder.conv2d(
      var_multiply_3_0, var_conv2d_21_filter_0,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_conv2d_20_1_y_0,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: 'Conv__213'
      }
    );
  
    
    const var_add_2__xeno_compat__1_0 = builder.add(
      var_conv__213_0,
      var_add_1__xeno_compat__1_0
    );
    
    const var_conv__216_0 = builder.conv2d(
      var_add_2__xeno_compat__1_0, var_conv2d_22_filter_0,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_conv2d_21_1_y_0,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: 'Conv__216'
      }
    );
  
    
    const var_add_7_0 = builder.add(
      var_conv__216_0,
      const_fold_opt__408
    );
    
    const var_relu6_7_0 = builder.clamp(
      var_add_7_0,
      {
        minValue: 0,
        maxValue: 6
      }
    );
    
    const var_mul_14_0 = builder.mul(
      var_mul_3_y_0,
      var_relu6_7_0
    );
    
    const var_mul_15_0 = builder.mul(
      var_conv__216_0,
      var_mul_14_0
    );
    
    const var_conv__219_0 = builder.conv2d(
      var_mul_15_0, const_fold_opt__395,
      {
        strides: [1, 1],
        padding: [2, 2, 2, 2],
        dilations: [1, 1],
        groups: 96,
        bias: var_depthwise_conv2d_6_y_0,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: 'Conv__219'
      }
    );
  
    
    const var_add_8_0 = builder.add(
      var_conv__219_0,
      const_fold_opt__408
    );
    
    const var_relu6_8_0 = builder.clamp(
      var_add_8_0,
      {
        minValue: 0,
        maxValue: 6
      }
    );
    
    const var_mul_16_0 = builder.mul(
      var_mul_3_y_0,
      var_relu6_8_0
    );
    
    const var_mul_17_0 = builder.mul(
      var_conv__219_0,
      var_mul_16_0
    );
    
    const var_global_average_pooling2d_4_0 = builder.averagePool2d(
      var_mul_17_0,
      { layout: 'nhwc' }
    );
    
    const var_conv__222_0 = builder.conv2d(
      var_global_average_pooling2d_4_0, var_conv2d_23_filter_0,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_conv2d_22_1_y_0,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: 'Conv__222'
      }
    );
  
    
    const var_re_lu_10_0 = builder.relu(
      var_conv__222_0,
      { label: 're_lu_10' }
    );
  
    
    const var_conv__223_0 = builder.conv2d(
      var_re_lu_10_0, var_conv2d_24_filter_0,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_conv2d_23_1_y_0,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: 'Conv__223'
      }
    );
  
    
    const var_activation_4_0 = builder.sigmoid(
      var_conv__223_0,
      { label: 'activation_4' }
    );
  
    
    const var_multiply_4_0 = builder.mul(
      var_mul_17_0,
      var_activation_4_0
    );
    
    const var_conv__224_0 = builder.conv2d(
      var_multiply_4_0, var_conv2d_25_filter_0,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_conv2d_24_1_y_0,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: 'Conv__224'
      }
    );
  
    
    const var_add_3__xeno_compat__1_0 = builder.add(
      var_conv__224_0,
      var_add_2__xeno_compat__1_0
    );
    
    const var_conv__227_0 = builder.conv2d(
      var_add_3__xeno_compat__1_0, var_conv2d_26_filter_0,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_conv2d_25_1_y_0,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: 'Conv__227'
      }
    );
  
    
    const var_add_9_0 = builder.add(
      var_conv__227_0,
      const_fold_opt__408
    );
    
    const var_relu6_9_0 = builder.clamp(
      var_add_9_0,
      {
        minValue: 0,
        maxValue: 6
      }
    );
    
    const var_mul_18_0 = builder.mul(
      var_mul_3_y_0,
      var_relu6_9_0
    );
    
    const var_mul_19_0 = builder.mul(
      var_conv__227_0,
      var_mul_18_0
    );
    
    const var_conv__230_0 = builder.conv2d(
      var_mul_19_0, const_fold_opt__407,
      {
        strides: [1, 1],
        padding: [2, 2, 2, 2],
        dilations: [1, 1],
        groups: 96,
        bias: var_depthwise_conv2d_7_y_0,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: 'Conv__230'
      }
    );
  
    
    const var_add_10_0 = builder.add(
      var_conv__230_0,
      const_fold_opt__408
    );
    
    const var_relu6_10_0 = builder.clamp(
      var_add_10_0,
      {
        minValue: 0,
        maxValue: 6
      }
    );
    
    const var_mul_20_0 = builder.mul(
      var_mul_3_y_0,
      var_relu6_10_0
    );
    
    const var_mul_21_0 = builder.mul(
      var_conv__230_0,
      var_mul_20_0
    );
    
    const var_global_average_pooling2d_5_0 = builder.averagePool2d(
      var_mul_21_0,
      { layout: 'nhwc' }
    );
    
    const var_conv__233_0 = builder.conv2d(
      var_global_average_pooling2d_5_0, var_conv2d_27_filter_0,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_conv2d_26_1_y_0,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: 'Conv__233'
      }
    );
  
    
    const var_re_lu_11_0 = builder.relu(
      var_conv__233_0,
      { label: 're_lu_11' }
    );
  
    
    const var_conv__234_0 = builder.conv2d(
      var_re_lu_11_0, var_conv2d_28_filter_0,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_conv2d_27_1_y_0,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: 'Conv__234'
      }
    );
  
    
    const var_activation_5_0 = builder.sigmoid(
      var_conv__234_0,
      { label: 'activation_5' }
    );
  
    
    const var_multiply_5_0 = builder.mul(
      var_mul_21_0,
      var_activation_5_0
    );
    
    const var_conv__235_0 = builder.conv2d(
      var_multiply_5_0, var_conv2d_29_filter_0,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_conv2d_28_1_y_0,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: 'Conv__235'
      }
    );
  
    
    const var_add_4__xeno_compat__1_0 = builder.add(
      var_conv__235_0,
      var_add_3__xeno_compat__1_0
    );
    
    const var_global_average_pooling2d_6_0 = builder.averagePool2d(
      var_add_4__xeno_compat__1_0,
      { layout: 'nhwc' }
    );
    
    const var_conv__238_0 = builder.conv2d(
      var_global_average_pooling2d_6_0, var_conv2d_31_filter_0,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_conv2d_30_1_y_0,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: 'Conv__238'
      }
    );
  
    
    const var_activation_6_0 = builder.sigmoid(
      var_conv__238_0,
      { label: 'activation_6' }
    );
  
    
    const var_conv__239_0 = builder.conv2d(
      var_add_4__xeno_compat__1_0, var_conv2d_30_filter_0,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_conv2d_29_1_y_0,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: 'Conv__239'
      }
    );
  
    
    const var_re_lu_12_0 = builder.relu(
      var_conv__239_0,
      { label: 're_lu_12' }
    );
  
    
    const var_multiply_6_0 = builder.mul(
      var_re_lu_12_0,
      var_activation_6_0
    );
    
    const var_289 = builder.resample2d(
      var_multiply_6_0,
      {
        mode: 'linear',
        sizes: undefined,
        scales: [2, 2],
        axes: [1, 2]
      }
    );
    
    const var_conv__240_0 = builder.conv2d(
      var_289, var_conv2d_32_filter_0,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_conv2d_31_1_y_0,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: 'Conv__240'
      }
    );
  
    
    const var_add_5__xeno_compat__1_0 = builder.add(
      var_add__xeno_compat__1_0,
      var_conv__240_0
    );
    
    const var_global_average_pooling2d_7_0 = builder.averagePool2d(
      var_add_5__xeno_compat__1_0,
      { layout: 'nhwc' }
    );
    
    const var_conv__243_0 = builder.conv2d(
      var_global_average_pooling2d_7_0, var_conv2d_33_filter_0,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_conv2d_32_1_y_0,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: 'Conv__243'
      }
    );
  
    
    const var_re_lu_13_0 = builder.relu(
      var_conv__243_0,
      { label: 're_lu_13' }
    );
  
    
    const var_conv__244_0 = builder.conv2d(
      var_re_lu_13_0, var_conv2d_34_filter_0,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_conv2d_33_1_y_0,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: 'Conv__244'
      }
    );
  
    
    const var_activation_7_0 = builder.sigmoid(
      var_conv__244_0,
      { label: 'activation_7' }
    );
  
    
    const var_multiply_7_0 = builder.mul(
      var_add__xeno_compat__1_0,
      var_activation_7_0
    );
    
    const var_add_6__xeno_compat__1_0 = builder.add(
      var_multiply_7_0,
      var_conv__240_0
    );
    
    const var_conv__245_0 = builder.conv2d(
      var_add_6__xeno_compat__1_0, var_conv2d_35_filter_0,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_conv2d_34_1_y_0,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: 'Conv__245'
      }
    );
  
    
    const var_re_lu_14_0 = builder.relu(
      var_conv__245_0,
      { label: 're_lu_14' }
    );
  
    
    const var_conv__248_0 = builder.conv2d(
      var_re_lu_14_0, const_fold_opt__394,
      {
        strides: [1, 1],
        padding: [1, 1, 1, 1],
        dilations: [1, 1],
        groups: 24,
        bias: var_depthwise_conv2d_8_y_0,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: 'Conv__248'
      }
    );
  
    
    const var_re_lu_15_0 = builder.relu(
      var_conv__248_0,
      { label: 're_lu_15' }
    );
  
    
    const var_add_7__xeno_compat__1_0 = builder.add(
      var_re_lu_14_0,
      var_re_lu_15_0
    );
    
    const var_290 = builder.resample2d(
      var_add_7__xeno_compat__1_0,
      {
        mode: 'linear',
        sizes: undefined,
        scales: [2, 2],
        axes: [1, 2]
      }
    );
    
    const var_conv__249_0 = builder.conv2d(
      var_290, var_conv2d_36_filter_0,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_conv2d_35_1_y_0,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: 'Conv__249'
      }
    );
  
    
    const var_add_8__xeno_compat__1_0 = builder.add(
      var_conv__167_0,
      var_conv__249_0
    );
    
    const var_global_average_pooling2d_8_0 = builder.averagePool2d(
      var_add_8__xeno_compat__1_0,
      { layout: 'nhwc' }
    );
    
    const var_conv__252_0 = builder.conv2d(
      var_global_average_pooling2d_8_0, var_conv2d_37_filter_0,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_conv2d_36_1_y_0,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: 'Conv__252'
      }
    );
  
    
    const var_re_lu_16_0 = builder.relu(
      var_conv__252_0,
      { label: 're_lu_16' }
    );
  
    
    const var_conv__253_0 = builder.conv2d(
      var_re_lu_16_0, var_conv2d_38_filter_0,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_conv2d_37_1_y_0,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: 'Conv__253'
      }
    );
  
    
    const var_activation_8_0 = builder.sigmoid(
      var_conv__253_0,
      { label: 'activation_8' }
    );
  
    
    const var_multiply_8_0 = builder.mul(
      var_conv__167_0,
      var_activation_8_0
    );
    
    const var_add_9__xeno_compat__1_0 = builder.add(
      var_multiply_8_0,
      var_conv__249_0
    );
    
    const var_conv__254_0 = builder.conv2d(
      var_add_9__xeno_compat__1_0, var_conv2d_39_filter_0,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_conv2d_38_1_y_0,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: 'Conv__254'
      }
    );
  
    
    const var_re_lu_17_0 = builder.relu(
      var_conv__254_0,
      { label: 're_lu_17' }
    );
  
    
    const var_conv__257_0 = builder.conv2d(
      var_re_lu_17_0, const_fold_opt__391,
      {
        strides: [1, 1],
        padding: [1, 1, 1, 1],
        dilations: [1, 1],
        groups: 16,
        bias: var_depthwise_conv2d_9_y_0,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: 'Conv__257'
      }
    );
  
    
    const var_re_lu_18_0 = builder.relu(
      var_conv__257_0,
      { label: 're_lu_18' }
    );
  
    
    const var_add_10__xeno_compat__1_0 = builder.add(
      var_re_lu_17_0,
      var_re_lu_18_0
    );
    
    const var_291 = builder.resample2d(
      var_add_10__xeno_compat__1_0,
      {
        mode: 'linear',
        sizes: undefined,
        scales: [2, 2],
        axes: [1, 2]
      }
    );
    
    const var_conv__258_0 = builder.conv2d(
      var_291, var_conv2d_40_filter_0,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_conv2d_39_1_y_0,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: 'Conv__258'
      }
    );
  
    
    const var_add_11__xeno_compat__1_0 = builder.add(
      var_mul_1_0,
      var_conv__258_0
    );
    
    const var_global_average_pooling2d_9_0 = builder.averagePool2d(
      var_add_11__xeno_compat__1_0,
      { layout: 'nhwc' }
    );
    
    const var_conv__261_0 = builder.conv2d(
      var_global_average_pooling2d_9_0, var_conv2d_41_filter_0,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_conv2d_40_1_y_0,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: 'Conv__261'
      }
    );
  
    
    const var_re_lu_19_0 = builder.relu(
      var_conv__261_0,
      { label: 're_lu_19' }
    );
  
    
    const var_conv__262_0 = builder.conv2d(
      var_re_lu_19_0, var_conv2d_42_filter_0,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_conv2d_41_1_y_0,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: 'Conv__262'
      }
    );
  
    
    const var_activation_9_0 = builder.sigmoid(
      var_conv__262_0,
      { label: 'activation_9' }
    );
  
    
    const var_multiply_9_0 = builder.mul(
      var_mul_1_0,
      var_activation_9_0
    );
    
    const var_add_12__xeno_compat__1_0 = builder.add(
      var_multiply_9_0,
      var_conv__258_0
    );
    
    const var_conv__263_0 = builder.conv2d(
      var_add_12__xeno_compat__1_0, var_conv2d_43_filter_0,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_conv2d_42_1_y_0,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: 'Conv__263'
      }
    );
  
    
    const var_re_lu_20_0 = builder.relu(
      var_conv__263_0,
      { label: 're_lu_20' }
    );
  
    
    const var_conv__266_0 = builder.conv2d(
      var_re_lu_20_0, const_fold_opt__389,
      {
        strides: [1, 1],
        padding: [1, 1, 1, 1],
        dilations: [1, 1],
        groups: 16,
        bias: var_depthwise_conv2d_10_y_0,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: 'Conv__266'
      }
    );
  
    
    const var_re_lu_21_0 = builder.relu(
      var_conv__266_0,
      { label: 're_lu_21' }
    );
  
    
    const var_add_13__xeno_compat__1_0 = builder.add(
      var_re_lu_20_0,
      var_re_lu_21_0
    );
    
    // ConvTranspose:
    // Input shape: [1,72,128,16]
    // Filter shape: [16,2,2,1]
    // NHWC mode: true
    // Using filterLayout: 'ohwi', inputLayout: 'nhwc'
    // ERROR: input channels (16) != filter input channels (1).
    // This usually means your weights file or model export is incorrect.
    const var_conv2d_transpose_0 = builder.convTranspose2d(
      var_add_13__xeno_compat__1_0, var_conv2d_transpose_filter_0,
      {
        strides: [2, 2],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        outputPadding: [0, 0],
        outputSizes: [144, 256],
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: 'conv2d_transpose'
      }
    );
  
    
    const var_conv2d_transpose_add_0 = builder.add(
      var_conv2d_transpose_0,
      const_fold_opt__402
    );
    
    const var_segment_back_raw_output___4_0 = builder.sigmoid(
      var_conv2d_transpose_add_0,
      { label: 'segment_back' }
    );
  
    
    const segment_back = builder.reshape(
      var_segment_back_raw_output___4_0,
      (() => {
        const shape = Array.from(new BigInt64Array(weights_array_buffer, 426104, 32 / BigInt64Array.BYTES_PER_ELEMENT), Number);
        // Calculate the concrete size for value -1.
        if (shape.includes(-1)) {
          const count = shape.filter(v => v === -1).length;
          if (count !== 1) {
            throw new Error('Only one -1 is allowed in reshape shape');
          }
          const totalInput = var_segment_back_raw_output___4_0.shape.reduce((a, b) => a * b, 1);
          const known = shape.reduce((a, b) => b === -1 ? a : a * b, 1);
          const idx = shape.indexOf(-1);
          shape[idx] = totalInput / known;
        }
        return shape;
      })()
    );


    // Build graph with all outputs
    
    this.graph_ = await builder.build({ 'segment_back': segment_back });

    // Create output tensors
    
    const segment_back_nhwc = builder.output('segment_back', { dataType: 'float32', shape: [1,256,1,144] });
    // Optionally, after inference, transpose back to NCHW if you want to present in ONNX order
    // const segment_back_nchw = builder.transpose(segment_back_nhwc, { permutation: [0, 3, 1, 2] });
    this.outputTensors_['segment_back'] = await this.context_.createTensor(
      { dataType: 'float32', shape: [1,256,1,144], readable: true }
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