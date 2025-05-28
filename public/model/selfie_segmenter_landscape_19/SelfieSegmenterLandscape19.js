// WebNN Code Generator
// Todo: NCHW, NHWC layouts for BatchNormalization, InstanceNormalization, Conv, ConvInteger, 
// QLinearConv, ConvTranspose, AveragePool, LpPool, MaxPool, MaxUnpool, GlobalAveragePool, 
// GlobalLpPool, GlobalMaxPool, LRN, GridSample, DepthToSpace, SpaceToDepth

export class SelfieSegmenterLandscape19 {

  constructor() {
    this.context_ = null;
    this.graph_ = null;
    this.inputTensors_ = {};
    this.outputTensors_ = {};
  }

  async build(options) {
    // Load weights ArrayBuffer from .bin file
    async function loadWeightsArrayBuffer() {
      const response = await fetch('weights.bin');
      if (!response.ok) {
          throw new Error('Failed to fetch weights: ' + response.statusText);
      }
      return await response.arrayBuffer();
    }

    const weights_array_buffer = await loadWeightsArrayBuffer();

    this.context_ = await navigator.ml.createContext(options);
    const builder = new MLGraphBuilder(this.context_);

    // Create graph input operands and tensors
    
    const input = builder.input('input', { dataType: 'float32', shape: [1,144,256,3] });
    this.inputTensors_['input'] = await this.context_.createTensor(
      { dataType: 'float32', shape: [1,144,256,3], writable: true }
    );

    // Create graph constant operands
    
    const var_conv2d_filter_0 = builder.constant(
      { dataType: 'float32', shape: [16,3,3,3] },
      new Float32Array(weights_array_buffer, 0, 1728 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [16] },
      new Float32Array(weights_array_buffer, 1728, 64 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const const_fold_opt__408 = builder.constant(
      { dataType: 'float32', shape: [1,1,1,1] },
      new Float32Array(weights_array_buffer, 329344, 4 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_292 = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0])
    );

    const var_293 = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([6])
    );

    const var_mul_3_y_0 = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array(weights_array_buffer, 329356, 4 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_2_filter_0 = builder.constant(
      { dataType: 'float32', shape: [16,16,1,1] },
      new Float32Array(weights_array_buffer, 1808, 1024 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_1_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [16] },
      new Float32Array(weights_array_buffer, 2832, 64 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const const_fold_opt__377 = builder.constant(
      { dataType: 'float32', shape: [16,1,3,3] },
      new Float32Array(weights_array_buffer, 2896, 576 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_depthwise_conv2d_y_0 = builder.constant(
      { dataType: 'float32', shape: [16] },
      new Float32Array(weights_array_buffer, 3472, 64 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_3_filter_0 = builder.constant(
      { dataType: 'float32', shape: [8,16,1,1] },
      new Float32Array(weights_array_buffer, 3536, 512 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_2_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [8] },
      new Float32Array(weights_array_buffer, 4048, 32 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_4_filter_0 = builder.constant(
      { dataType: 'float32', shape: [16,8,1,1] },
      new Float32Array(weights_array_buffer, 4080, 512 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_3_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [16] },
      new Float32Array(weights_array_buffer, 4592, 64 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_5_filter_0 = builder.constant(
      { dataType: 'float32', shape: [16,16,1,1] },
      new Float32Array(weights_array_buffer, 4656, 1024 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_4_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [16] },
      new Float32Array(weights_array_buffer, 5680, 64 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_6_filter_0 = builder.constant(
      { dataType: 'float32', shape: [72,16,1,1] },
      new Float32Array(weights_array_buffer, 5744, 4608 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_5_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [72] },
      new Float32Array(weights_array_buffer, 10352, 288 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const const_fold_opt__387 = builder.constant(
      { dataType: 'float32', shape: [72,1,3,3] },
      new Float32Array(weights_array_buffer, 10640, 2592 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_depthwise_conv2d_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [72] },
      new Float32Array(weights_array_buffer, 13232, 288 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_7_filter_0 = builder.constant(
      { dataType: 'float32', shape: [24,72,1,1] },
      new Float32Array(weights_array_buffer, 13520, 6912 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_6_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [24] },
      new Float32Array(weights_array_buffer, 20432, 96 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_8_filter_0 = builder.constant(
      { dataType: 'float32', shape: [88,24,1,1] },
      new Float32Array(weights_array_buffer, 20528, 8448 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_7_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [88] },
      new Float32Array(weights_array_buffer, 28976, 352 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const const_fold_opt__385 = builder.constant(
      { dataType: 'float32', shape: [88,1,3,3] },
      new Float32Array(weights_array_buffer, 29328, 3168 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_depthwise_conv2d_2_y_0 = builder.constant(
      { dataType: 'float32', shape: [88] },
      new Float32Array(weights_array_buffer, 32496, 352 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_9_filter_0 = builder.constant(
      { dataType: 'float32', shape: [24,88,1,1] },
      new Float32Array(weights_array_buffer, 32848, 8448 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_8_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [24] },
      new Float32Array(weights_array_buffer, 41296, 96 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_10_filter_0 = builder.constant(
      { dataType: 'float32', shape: [96,24,1,1] },
      new Float32Array(weights_array_buffer, 41392, 9216 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_9_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [96] },
      new Float32Array(weights_array_buffer, 50608, 384 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_294 = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0])
    );

    const var_295 = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([6])
    );

    const const_fold_opt__383 = builder.constant(
      { dataType: 'float32', shape: [96,1,5,5] },
      new Float32Array(weights_array_buffer, 51008, 9600 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_depthwise_conv2d_3_y_0 = builder.constant(
      { dataType: 'float32', shape: [96] },
      new Float32Array(weights_array_buffer, 60608, 384 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_296 = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0])
    );

    const var_297 = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([6])
    );

    const var_conv2d_11_filter_0 = builder.constant(
      { dataType: 'float32', shape: [24,96,1,1] },
      new Float32Array(weights_array_buffer, 61008, 9216 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_10_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [24] },
      new Float32Array(weights_array_buffer, 70224, 96 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_12_filter_0 = builder.constant(
      { dataType: 'float32', shape: [96,24,1,1] },
      new Float32Array(weights_array_buffer, 70320, 9216 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_11_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [96] },
      new Float32Array(weights_array_buffer, 79536, 384 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_13_filter_0 = builder.constant(
      { dataType: 'float32', shape: [32,96,1,1] },
      new Float32Array(weights_array_buffer, 79920, 12288 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_12_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [32] },
      new Float32Array(weights_array_buffer, 92208, 128 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_14_filter_0 = builder.constant(
      { dataType: 'float32', shape: [128,32,1,1] },
      new Float32Array(weights_array_buffer, 92336, 16384 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_13_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [128] },
      new Float32Array(weights_array_buffer, 108720, 512 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_298 = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0])
    );

    const var_299 = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([6])
    );

    const const_fold_opt__381 = builder.constant(
      { dataType: 'float32', shape: [128,1,5,5] },
      new Float32Array(weights_array_buffer, 109248, 12800 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_depthwise_conv2d_4_y_0 = builder.constant(
      { dataType: 'float32', shape: [128] },
      new Float32Array(weights_array_buffer, 122048, 512 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_300 = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0])
    );

    const var_301 = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([6])
    );

    const var_conv2d_15_filter_0 = builder.constant(
      { dataType: 'float32', shape: [32,128,1,1] },
      new Float32Array(weights_array_buffer, 122576, 16384 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_14_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [32] },
      new Float32Array(weights_array_buffer, 138960, 128 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_16_filter_0 = builder.constant(
      { dataType: 'float32', shape: [128,32,1,1] },
      new Float32Array(weights_array_buffer, 139088, 16384 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_15_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [128] },
      new Float32Array(weights_array_buffer, 155472, 512 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_17_filter_0 = builder.constant(
      { dataType: 'float32', shape: [32,128,1,1] },
      new Float32Array(weights_array_buffer, 155984, 16384 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_16_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [32] },
      new Float32Array(weights_array_buffer, 172368, 128 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_18_filter_0 = builder.constant(
      { dataType: 'float32', shape: [128,32,1,1] },
      new Float32Array(weights_array_buffer, 172496, 16384 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_17_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [128] },
      new Float32Array(weights_array_buffer, 188880, 512 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_302 = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0])
    );

    const var_303 = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([6])
    );

    const const_fold_opt__379 = builder.constant(
      { dataType: 'float32', shape: [128,1,5,5] },
      new Float32Array(weights_array_buffer, 189408, 12800 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_depthwise_conv2d_5_y_0 = builder.constant(
      { dataType: 'float32', shape: [128] },
      new Float32Array(weights_array_buffer, 202208, 512 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_304 = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0])
    );

    const var_305 = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([6])
    );

    const var_conv2d_19_filter_0 = builder.constant(
      { dataType: 'float32', shape: [32,128,1,1] },
      new Float32Array(weights_array_buffer, 202736, 16384 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_18_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [32] },
      new Float32Array(weights_array_buffer, 219120, 128 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_20_filter_0 = builder.constant(
      { dataType: 'float32', shape: [128,32,1,1] },
      new Float32Array(weights_array_buffer, 219248, 16384 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_19_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [128] },
      new Float32Array(weights_array_buffer, 235632, 512 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_21_filter_0 = builder.constant(
      { dataType: 'float32', shape: [32,128,1,1] },
      new Float32Array(weights_array_buffer, 236144, 16384 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_20_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [32] },
      new Float32Array(weights_array_buffer, 252528, 128 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_22_filter_0 = builder.constant(
      { dataType: 'float32', shape: [96,32,1,1] },
      new Float32Array(weights_array_buffer, 252656, 12288 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_21_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [96] },
      new Float32Array(weights_array_buffer, 264944, 384 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_306 = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0])
    );

    const var_307 = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([6])
    );

    const const_fold_opt__395 = builder.constant(
      { dataType: 'float32', shape: [96,1,5,5] },
      new Float32Array(weights_array_buffer, 265344, 9600 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_depthwise_conv2d_6_y_0 = builder.constant(
      { dataType: 'float32', shape: [96] },
      new Float32Array(weights_array_buffer, 274944, 384 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_308 = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0])
    );

    const var_309 = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([6])
    );

    const var_conv2d_23_filter_0 = builder.constant(
      { dataType: 'float32', shape: [24,96,1,1] },
      new Float32Array(weights_array_buffer, 275344, 9216 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_22_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [24] },
      new Float32Array(weights_array_buffer, 284560, 96 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_24_filter_0 = builder.constant(
      { dataType: 'float32', shape: [96,24,1,1] },
      new Float32Array(weights_array_buffer, 284656, 9216 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_23_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [96] },
      new Float32Array(weights_array_buffer, 293872, 384 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_25_filter_0 = builder.constant(
      { dataType: 'float32', shape: [32,96,1,1] },
      new Float32Array(weights_array_buffer, 294256, 12288 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_24_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [32] },
      new Float32Array(weights_array_buffer, 306544, 128 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_26_filter_0 = builder.constant(
      { dataType: 'float32', shape: [96,32,1,1] },
      new Float32Array(weights_array_buffer, 306672, 12288 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_25_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [96] },
      new Float32Array(weights_array_buffer, 318960, 384 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_310 = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0])
    );

    const var_311 = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([6])
    );

    const const_fold_opt__407 = builder.constant(
      { dataType: 'float32', shape: [96,1,5,5] },
      new Float32Array(weights_array_buffer, 319360, 9600 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_depthwise_conv2d_7_y_0 = builder.constant(
      { dataType: 'float32', shape: [96] },
      new Float32Array(weights_array_buffer, 328960, 384 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_312 = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0])
    );

    const var_313 = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([6])
    );

    const var_conv2d_27_filter_0 = builder.constant(
      { dataType: 'float32', shape: [24,96,1,1] },
      new Float32Array(weights_array_buffer, 329360, 9216 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_26_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [24] },
      new Float32Array(weights_array_buffer, 338576, 96 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_28_filter_0 = builder.constant(
      { dataType: 'float32', shape: [96,24,1,1] },
      new Float32Array(weights_array_buffer, 338672, 9216 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_27_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [96] },
      new Float32Array(weights_array_buffer, 347888, 384 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_29_filter_0 = builder.constant(
      { dataType: 'float32', shape: [32,96,1,1] },
      new Float32Array(weights_array_buffer, 348272, 12288 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_28_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [32] },
      new Float32Array(weights_array_buffer, 360560, 128 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_31_filter_0 = builder.constant(
      { dataType: 'float32', shape: [128,32,1,1] },
      new Float32Array(weights_array_buffer, 360688, 16384 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_30_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [128] },
      new Float32Array(weights_array_buffer, 377072, 512 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_30_filter_0 = builder.constant(
      { dataType: 'float32', shape: [128,32,1,1] },
      new Float32Array(weights_array_buffer, 377584, 16384 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_29_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [128] },
      new Float32Array(weights_array_buffer, 393968, 512 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_314 = builder.constant(
      { dataType: 'float32', shape: [8] },
      new Float32Array([0, 0, 0, 0, 1, 1, 1, 1])
    );

    const scales__125 = builder.constant(
      { dataType: 'float32', shape: [4] },
      new Float32Array([1, 1, 2, 2])
    );

    const var_conv2d_32_filter_0 = builder.constant(
      { dataType: 'float32', shape: [24,128,1,1] },
      new Float32Array(weights_array_buffer, 394528, 12288 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_31_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [24] },
      new Float32Array(weights_array_buffer, 406816, 96 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_33_filter_0 = builder.constant(
      { dataType: 'float32', shape: [24,24,1,1] },
      new Float32Array(weights_array_buffer, 406912, 2304 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_32_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [24] },
      new Float32Array(weights_array_buffer, 409216, 96 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_34_filter_0 = builder.constant(
      { dataType: 'float32', shape: [24,24,1,1] },
      new Float32Array(weights_array_buffer, 409312, 2304 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_33_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [24] },
      new Float32Array(weights_array_buffer, 411616, 96 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_35_filter_0 = builder.constant(
      { dataType: 'float32', shape: [24,24,1,1] },
      new Float32Array(weights_array_buffer, 411712, 2304 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_34_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [24] },
      new Float32Array(weights_array_buffer, 414016, 96 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const const_fold_opt__394 = builder.constant(
      { dataType: 'float32', shape: [24,1,3,3] },
      new Float32Array(weights_array_buffer, 414112, 864 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_depthwise_conv2d_8_y_0 = builder.constant(
      { dataType: 'float32', shape: [24] },
      new Float32Array(weights_array_buffer, 414976, 96 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_315 = builder.constant(
      { dataType: 'float32', shape: [8] },
      new Float32Array([0, 0, 0, 0, 1, 1, 1, 1])
    );

    const var_conv2d_36_filter_0 = builder.constant(
      { dataType: 'float32', shape: [16,24,1,1] },
      new Float32Array(weights_array_buffer, 415120, 1536 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_35_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [16] },
      new Float32Array(weights_array_buffer, 416656, 64 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_37_filter_0 = builder.constant(
      { dataType: 'float32', shape: [16,16,1,1] },
      new Float32Array(weights_array_buffer, 416720, 1024 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_36_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [16] },
      new Float32Array(weights_array_buffer, 417744, 64 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_38_filter_0 = builder.constant(
      { dataType: 'float32', shape: [16,16,1,1] },
      new Float32Array(weights_array_buffer, 417808, 1024 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_37_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [16] },
      new Float32Array(weights_array_buffer, 418832, 64 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_39_filter_0 = builder.constant(
      { dataType: 'float32', shape: [16,16,1,1] },
      new Float32Array(weights_array_buffer, 418896, 1024 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_38_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [16] },
      new Float32Array(weights_array_buffer, 419920, 64 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const const_fold_opt__391 = builder.constant(
      { dataType: 'float32', shape: [16,1,3,3] },
      new Float32Array(weights_array_buffer, 419984, 576 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_depthwise_conv2d_9_y_0 = builder.constant(
      { dataType: 'float32', shape: [16] },
      new Float32Array(weights_array_buffer, 420560, 64 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_316 = builder.constant(
      { dataType: 'float32', shape: [8] },
      new Float32Array([0, 0, 0, 0, 1, 1, 1, 1])
    );

    const var_conv2d_40_filter_0 = builder.constant(
      { dataType: 'float32', shape: [16,16,1,1] },
      new Float32Array(weights_array_buffer, 420672, 1024 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_39_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [16] },
      new Float32Array(weights_array_buffer, 421696, 64 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_41_filter_0 = builder.constant(
      { dataType: 'float32', shape: [16,16,1,1] },
      new Float32Array(weights_array_buffer, 421760, 1024 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_40_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [16] },
      new Float32Array(weights_array_buffer, 422784, 64 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_42_filter_0 = builder.constant(
      { dataType: 'float32', shape: [16,16,1,1] },
      new Float32Array(weights_array_buffer, 422848, 1024 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_41_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [16] },
      new Float32Array(weights_array_buffer, 423872, 64 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_43_filter_0 = builder.constant(
      { dataType: 'float32', shape: [16,16,1,1] },
      new Float32Array(weights_array_buffer, 423936, 1024 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_42_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [16] },
      new Float32Array(weights_array_buffer, 424960, 64 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const const_fold_opt__389 = builder.constant(
      { dataType: 'float32', shape: [16,1,3,3] },
      new Float32Array(weights_array_buffer, 425024, 576 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_depthwise_conv2d_10_y_0 = builder.constant(
      { dataType: 'float32', shape: [16] },
      new Float32Array(weights_array_buffer, 425600, 64 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_transpose_filter_0 = builder.constant(
      { dataType: 'float32', shape: [16,1,2,2] },
      new Float32Array(weights_array_buffer, 425664, 256 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const const_fold_opt__402 = builder.constant(
      { dataType: 'float32', shape: [1,1,1,1] },
      new Float32Array(weights_array_buffer, 425920, 4 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const new_shape__369 = builder.constant(
      { dataType: 'int64', shape: [4] },
      new BigInt64Array(weights_array_buffer.slice(425924, 425956))
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
        filterLayout: 'oihw',
        inputLayout: 'nchw'
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
        filterLayout: 'oihw',
        inputLayout: 'nchw'
      }
    );
    
    const var_re_lu_0 = builder.relu(
      var_conv__161_0
    );
    
    const var_conv__162_0 = builder.conv2d(
      var_re_lu_0, const_fold_opt__377,
      {
        strides: [2, 2],
        padding: [0, 1, 0, 1],
        dilations: [1, 1],
        groups: 16,
        bias: var_depthwise_conv2d_y_0,
        filterLayout: 'oihw',
        inputLayout: 'nchw'
      }
    );
    
    const var_re_lu_1_0 = builder.relu(
      var_conv__162_0
    );
    
    const var_global_average_pooling2d_0 = builder.averagePool2d(
      var_re_lu_1_0
    );
    
    const var_conv__165_0 = builder.conv2d(
      var_global_average_pooling2d_0, var_conv2d_3_filter_0,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_conv2d_2_1_y_0,
        filterLayout: 'oihw',
        inputLayout: 'nchw'
      }
    );
    
    const var_re_lu_2_0 = builder.relu(
      var_conv__165_0
    );
    
    const var_conv__166_0 = builder.conv2d(
      var_re_lu_2_0, var_conv2d_4_filter_0,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_conv2d_3_1_y_0,
        filterLayout: 'oihw',
        inputLayout: 'nchw'
      }
    );
    
    const var_activation_0 = builder.sigmoid(
      var_conv__166_0
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
        filterLayout: 'oihw',
        inputLayout: 'nchw'
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
        filterLayout: 'oihw',
        inputLayout: 'nchw'
      }
    );
    
    const var_re_lu_3_0 = builder.relu(
      var_conv__171_0
    );
    
    const var_conv__172_0 = builder.conv2d(
      var_re_lu_3_0, const_fold_opt__387,
      {
        strides: [2, 2],
        padding: [0, 1, 0, 1],
        dilations: [1, 1],
        groups: 72,
        bias: var_depthwise_conv2d_1_y_0,
        filterLayout: 'oihw',
        inputLayout: 'nchw'
      }
    );
    
    const var_re_lu_4_0 = builder.relu(
      var_conv__172_0
    );
    
    const var_conv__173_0 = builder.conv2d(
      var_re_lu_4_0, var_conv2d_7_filter_0,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_conv2d_6_1_y_0,
        filterLayout: 'oihw',
        inputLayout: 'nchw'
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
        filterLayout: 'oihw',
        inputLayout: 'nchw'
      }
    );
    
    const var_re_lu_5_0 = builder.relu(
      var_conv__176_0
    );
    
    const var_conv__177_0 = builder.conv2d(
      var_re_lu_5_0, const_fold_opt__385,
      {
        strides: [1, 1],
        padding: [1, 1, 1, 1],
        dilations: [1, 1],
        groups: 88,
        bias: var_depthwise_conv2d_2_y_0,
        filterLayout: 'oihw',
        inputLayout: 'nchw'
      }
    );
    
    const var_re_lu_6_0 = builder.relu(
      var_conv__177_0
    );
    
    const var_conv__178_0 = builder.conv2d(
      var_re_lu_6_0, var_conv2d_9_filter_0,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_conv2d_8_1_y_0,
        filterLayout: 'oihw',
        inputLayout: 'nchw'
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
        filterLayout: 'oihw',
        inputLayout: 'nchw'
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
        filterLayout: 'oihw',
        inputLayout: 'nchw'
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
      var_mul_5_0
    );
    
    const var_conv__189_0 = builder.conv2d(
      var_global_average_pooling2d_1_0, var_conv2d_11_filter_0,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_conv2d_10_1_y_0,
        filterLayout: 'oihw',
        inputLayout: 'nchw'
      }
    );
    
    const var_re_lu_7_0 = builder.relu(
      var_conv__189_0
    );
    
    const var_conv__190_0 = builder.conv2d(
      var_re_lu_7_0, var_conv2d_12_filter_0,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_conv2d_11_1_y_0,
        filterLayout: 'oihw',
        inputLayout: 'nchw'
      }
    );
    
    const var_activation_1_0 = builder.sigmoid(
      var_conv__190_0
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
        filterLayout: 'oihw',
        inputLayout: 'nchw'
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
        filterLayout: 'oihw',
        inputLayout: 'nchw'
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
        filterLayout: 'oihw',
        inputLayout: 'nchw'
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
      var_mul_9_0
    );
    
    const var_conv__200_0 = builder.conv2d(
      var_global_average_pooling2d_2_0, var_conv2d_15_filter_0,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_conv2d_14_1_y_0,
        filterLayout: 'oihw',
        inputLayout: 'nchw'
      }
    );
    
    const var_re_lu_8_0 = builder.relu(
      var_conv__200_0
    );
    
    const var_conv__201_0 = builder.conv2d(
      var_re_lu_8_0, var_conv2d_16_filter_0,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_conv2d_15_1_y_0,
        filterLayout: 'oihw',
        inputLayout: 'nchw'
      }
    );
    
    const var_activation_2_0 = builder.sigmoid(
      var_conv__201_0
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
        filterLayout: 'oihw',
        inputLayout: 'nchw'
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
        filterLayout: 'oihw',
        inputLayout: 'nchw'
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
        filterLayout: 'oihw',
        inputLayout: 'nchw'
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
      var_mul_13_0
    );
    
    const var_conv__211_0 = builder.conv2d(
      var_global_average_pooling2d_3_0, var_conv2d_19_filter_0,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_conv2d_18_1_y_0,
        filterLayout: 'oihw',
        inputLayout: 'nchw'
      }
    );
    
    const var_re_lu_9_0 = builder.relu(
      var_conv__211_0
    );
    
    const var_conv__212_0 = builder.conv2d(
      var_re_lu_9_0, var_conv2d_20_filter_0,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_conv2d_19_1_y_0,
        filterLayout: 'oihw',
        inputLayout: 'nchw'
      }
    );
    
    const var_activation_3_0 = builder.sigmoid(
      var_conv__212_0
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
        filterLayout: 'oihw',
        inputLayout: 'nchw'
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
        filterLayout: 'oihw',
        inputLayout: 'nchw'
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
        filterLayout: 'oihw',
        inputLayout: 'nchw'
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
      var_mul_17_0
    );
    
    const var_conv__222_0 = builder.conv2d(
      var_global_average_pooling2d_4_0, var_conv2d_23_filter_0,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_conv2d_22_1_y_0,
        filterLayout: 'oihw',
        inputLayout: 'nchw'
      }
    );
    
    const var_re_lu_10_0 = builder.relu(
      var_conv__222_0
    );
    
    const var_conv__223_0 = builder.conv2d(
      var_re_lu_10_0, var_conv2d_24_filter_0,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_conv2d_23_1_y_0,
        filterLayout: 'oihw',
        inputLayout: 'nchw'
      }
    );
    
    const var_activation_4_0 = builder.sigmoid(
      var_conv__223_0
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
        filterLayout: 'oihw',
        inputLayout: 'nchw'
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
        filterLayout: 'oihw',
        inputLayout: 'nchw'
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
        filterLayout: 'oihw',
        inputLayout: 'nchw'
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
      var_mul_21_0
    );
    
    const var_conv__233_0 = builder.conv2d(
      var_global_average_pooling2d_5_0, var_conv2d_27_filter_0,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_conv2d_26_1_y_0,
        filterLayout: 'oihw',
        inputLayout: 'nchw'
      }
    );
    
    const var_re_lu_11_0 = builder.relu(
      var_conv__233_0
    );
    
    const var_conv__234_0 = builder.conv2d(
      var_re_lu_11_0, var_conv2d_28_filter_0,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_conv2d_27_1_y_0,
        filterLayout: 'oihw',
        inputLayout: 'nchw'
      }
    );
    
    const var_activation_5_0 = builder.sigmoid(
      var_conv__234_0
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
        filterLayout: 'oihw',
        inputLayout: 'nchw'
      }
    );
    
    const var_add_4__xeno_compat__1_0 = builder.add(
      var_conv__235_0,
      var_add_3__xeno_compat__1_0
    );
    
    const var_global_average_pooling2d_6_0 = builder.averagePool2d(
      var_add_4__xeno_compat__1_0
    );
    
    const var_conv__238_0 = builder.conv2d(
      var_global_average_pooling2d_6_0, var_conv2d_31_filter_0,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_conv2d_30_1_y_0,
        filterLayout: 'oihw',
        inputLayout: 'nchw'
      }
    );
    
    const var_activation_6_0 = builder.sigmoid(
      var_conv__238_0
    );
    
    const var_conv__239_0 = builder.conv2d(
      var_add_4__xeno_compat__1_0, var_conv2d_30_filter_0,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_conv2d_29_1_y_0,
        filterLayout: 'oihw',
        inputLayout: 'nchw'
      }
    );
    
    const var_re_lu_12_0 = builder.relu(
      var_conv__239_0
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
        axes: [2, 3]
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
        filterLayout: 'oihw',
        inputLayout: 'nchw'
      }
    );
    
    const var_add_5__xeno_compat__1_0 = builder.add(
      var_add__xeno_compat__1_0,
      var_conv__240_0
    );
    
    const var_global_average_pooling2d_7_0 = builder.averagePool2d(
      var_add_5__xeno_compat__1_0
    );
    
    const var_conv__243_0 = builder.conv2d(
      var_global_average_pooling2d_7_0, var_conv2d_33_filter_0,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_conv2d_32_1_y_0,
        filterLayout: 'oihw',
        inputLayout: 'nchw'
      }
    );
    
    const var_re_lu_13_0 = builder.relu(
      var_conv__243_0
    );
    
    const var_conv__244_0 = builder.conv2d(
      var_re_lu_13_0, var_conv2d_34_filter_0,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_conv2d_33_1_y_0,
        filterLayout: 'oihw',
        inputLayout: 'nchw'
      }
    );
    
    const var_activation_7_0 = builder.sigmoid(
      var_conv__244_0
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
        filterLayout: 'oihw',
        inputLayout: 'nchw'
      }
    );
    
    const var_re_lu_14_0 = builder.relu(
      var_conv__245_0
    );
    
    const var_conv__248_0 = builder.conv2d(
      var_re_lu_14_0, const_fold_opt__394,
      {
        strides: [1, 1],
        padding: [1, 1, 1, 1],
        dilations: [1, 1],
        groups: 24,
        bias: var_depthwise_conv2d_8_y_0,
        filterLayout: 'oihw',
        inputLayout: 'nchw'
      }
    );
    
    const var_re_lu_15_0 = builder.relu(
      var_conv__248_0
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
        axes: [2, 3]
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
        filterLayout: 'oihw',
        inputLayout: 'nchw'
      }
    );
    
    const var_add_8__xeno_compat__1_0 = builder.add(
      var_conv__167_0,
      var_conv__249_0
    );
    
    const var_global_average_pooling2d_8_0 = builder.averagePool2d(
      var_add_8__xeno_compat__1_0
    );
    
    const var_conv__252_0 = builder.conv2d(
      var_global_average_pooling2d_8_0, var_conv2d_37_filter_0,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_conv2d_36_1_y_0,
        filterLayout: 'oihw',
        inputLayout: 'nchw'
      }
    );
    
    const var_re_lu_16_0 = builder.relu(
      var_conv__252_0
    );
    
    const var_conv__253_0 = builder.conv2d(
      var_re_lu_16_0, var_conv2d_38_filter_0,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_conv2d_37_1_y_0,
        filterLayout: 'oihw',
        inputLayout: 'nchw'
      }
    );
    
    const var_activation_8_0 = builder.sigmoid(
      var_conv__253_0
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
        filterLayout: 'oihw',
        inputLayout: 'nchw'
      }
    );
    
    const var_re_lu_17_0 = builder.relu(
      var_conv__254_0
    );
    
    const var_conv__257_0 = builder.conv2d(
      var_re_lu_17_0, const_fold_opt__391,
      {
        strides: [1, 1],
        padding: [1, 1, 1, 1],
        dilations: [1, 1],
        groups: 16,
        bias: var_depthwise_conv2d_9_y_0,
        filterLayout: 'oihw',
        inputLayout: 'nchw'
      }
    );
    
    const var_re_lu_18_0 = builder.relu(
      var_conv__257_0
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
        axes: [2, 3]
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
        filterLayout: 'oihw',
        inputLayout: 'nchw'
      }
    );
    
    const var_add_11__xeno_compat__1_0 = builder.add(
      var_mul_1_0,
      var_conv__258_0
    );
    
    const var_global_average_pooling2d_9_0 = builder.averagePool2d(
      var_add_11__xeno_compat__1_0
    );
    
    const var_conv__261_0 = builder.conv2d(
      var_global_average_pooling2d_9_0, var_conv2d_41_filter_0,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_conv2d_40_1_y_0,
        filterLayout: 'oihw',
        inputLayout: 'nchw'
      }
    );
    
    const var_re_lu_19_0 = builder.relu(
      var_conv__261_0
    );
    
    const var_conv__262_0 = builder.conv2d(
      var_re_lu_19_0, var_conv2d_42_filter_0,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_conv2d_41_1_y_0,
        filterLayout: 'oihw',
        inputLayout: 'nchw'
      }
    );
    
    const var_activation_9_0 = builder.sigmoid(
      var_conv__262_0
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
        filterLayout: 'oihw',
        inputLayout: 'nchw'
      }
    );
    
    const var_re_lu_20_0 = builder.relu(
      var_conv__263_0
    );
    
    const var_conv__266_0 = builder.conv2d(
      var_re_lu_20_0, const_fold_opt__389,
      {
        strides: [1, 1],
        padding: [1, 1, 1, 1],
        dilations: [1, 1],
        groups: 16,
        bias: var_depthwise_conv2d_10_y_0,
        filterLayout: 'oihw',
        inputLayout: 'nchw'
      }
    );
    
    const var_re_lu_21_0 = builder.relu(
      var_conv__266_0
    );
    
    const var_add_13__xeno_compat__1_0 = builder.add(
      var_re_lu_20_0,
      var_re_lu_21_0
    );
    
    const var_conv2d_transpose_0 = builder.convTranspose2d(
      var_add_13__xeno_compat__1_0, var_conv2d_transpose_filter_0,
      {
        strides: [2, 2],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        outputSizes: [144, 256]
      }
    );
    
    const var_conv2d_transpose_add_0 = builder.add(
      var_conv2d_transpose_0,
      const_fold_opt__402
    );
    
    const var_segment_back_raw_output___4_0 = builder.sigmoid(
      var_conv2d_transpose_add_0
    );
    
    const segment_back = builder.reshape(
      var_segment_back_raw_output___4_0,
      (() => {
        const shape = Array.from(new BigInt64Array(weights_array_buffer.slice(425924, 425956)), Number);
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
    
    this.outputTensors_['segment_back'] = await this.context_.createTensor(
      { dataType: 'float32', shape: [1,144,256,1], readable: true }
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