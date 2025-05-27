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
      new Float32Array(weights_array_buffer, 329264, 4 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_mul_3_y_0 = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array(weights_array_buffer, 329268, 4 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_2_filter_0 = builder.constant(
      { dataType: 'float32', shape: [16,16,1,1] },
      new Float32Array(weights_array_buffer, 1800, 1024 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_1_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [16] },
      new Float32Array(weights_array_buffer, 2824, 64 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const const_fold_opt__377 = builder.constant(
      { dataType: 'float32', shape: [16,1,3,3] },
      new Float32Array(weights_array_buffer, 2888, 576 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_depthwise_conv2d_y_0 = builder.constant(
      { dataType: 'float32', shape: [16] },
      new Float32Array(weights_array_buffer, 3464, 64 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_3_filter_0 = builder.constant(
      { dataType: 'float32', shape: [8,16,1,1] },
      new Float32Array(weights_array_buffer, 3528, 512 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_2_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [8] },
      new Float32Array(weights_array_buffer, 4040, 32 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_4_filter_0 = builder.constant(
      { dataType: 'float32', shape: [16,8,1,1] },
      new Float32Array(weights_array_buffer, 4072, 512 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_3_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [16] },
      new Float32Array(weights_array_buffer, 4584, 64 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_5_filter_0 = builder.constant(
      { dataType: 'float32', shape: [16,16,1,1] },
      new Float32Array(weights_array_buffer, 4648, 1024 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_4_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [16] },
      new Float32Array(weights_array_buffer, 5672, 64 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_6_filter_0 = builder.constant(
      { dataType: 'float32', shape: [72,16,1,1] },
      new Float32Array(weights_array_buffer, 5736, 4608 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_5_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [72] },
      new Float32Array(weights_array_buffer, 10344, 288 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const const_fold_opt__387 = builder.constant(
      { dataType: 'float32', shape: [72,1,3,3] },
      new Float32Array(weights_array_buffer, 10632, 2592 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_depthwise_conv2d_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [72] },
      new Float32Array(weights_array_buffer, 13224, 288 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_7_filter_0 = builder.constant(
      { dataType: 'float32', shape: [24,72,1,1] },
      new Float32Array(weights_array_buffer, 13512, 6912 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_6_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [24] },
      new Float32Array(weights_array_buffer, 20424, 96 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_8_filter_0 = builder.constant(
      { dataType: 'float32', shape: [88,24,1,1] },
      new Float32Array(weights_array_buffer, 20520, 8448 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_7_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [88] },
      new Float32Array(weights_array_buffer, 28968, 352 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const const_fold_opt__385 = builder.constant(
      { dataType: 'float32', shape: [88,1,3,3] },
      new Float32Array(weights_array_buffer, 29320, 3168 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_depthwise_conv2d_2_y_0 = builder.constant(
      { dataType: 'float32', shape: [88] },
      new Float32Array(weights_array_buffer, 32488, 352 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_9_filter_0 = builder.constant(
      { dataType: 'float32', shape: [24,88,1,1] },
      new Float32Array(weights_array_buffer, 32840, 8448 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_8_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [24] },
      new Float32Array(weights_array_buffer, 41288, 96 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_10_filter_0 = builder.constant(
      { dataType: 'float32', shape: [96,24,1,1] },
      new Float32Array(weights_array_buffer, 41384, 9216 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_9_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [96] },
      new Float32Array(weights_array_buffer, 50600, 384 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const const_fold_opt__383 = builder.constant(
      { dataType: 'float32', shape: [96,1,5,5] },
      new Float32Array(weights_array_buffer, 50992, 9600 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_depthwise_conv2d_3_y_0 = builder.constant(
      { dataType: 'float32', shape: [96] },
      new Float32Array(weights_array_buffer, 60592, 384 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_11_filter_0 = builder.constant(
      { dataType: 'float32', shape: [24,96,1,1] },
      new Float32Array(weights_array_buffer, 60984, 9216 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_10_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [24] },
      new Float32Array(weights_array_buffer, 70200, 96 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_12_filter_0 = builder.constant(
      { dataType: 'float32', shape: [96,24,1,1] },
      new Float32Array(weights_array_buffer, 70296, 9216 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_11_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [96] },
      new Float32Array(weights_array_buffer, 79512, 384 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_13_filter_0 = builder.constant(
      { dataType: 'float32', shape: [32,96,1,1] },
      new Float32Array(weights_array_buffer, 79896, 12288 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_12_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [32] },
      new Float32Array(weights_array_buffer, 92184, 128 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_14_filter_0 = builder.constant(
      { dataType: 'float32', shape: [128,32,1,1] },
      new Float32Array(weights_array_buffer, 92312, 16384 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_13_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [128] },
      new Float32Array(weights_array_buffer, 108696, 512 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const const_fold_opt__381 = builder.constant(
      { dataType: 'float32', shape: [128,1,5,5] },
      new Float32Array(weights_array_buffer, 109216, 12800 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_depthwise_conv2d_4_y_0 = builder.constant(
      { dataType: 'float32', shape: [128] },
      new Float32Array(weights_array_buffer, 122016, 512 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_15_filter_0 = builder.constant(
      { dataType: 'float32', shape: [32,128,1,1] },
      new Float32Array(weights_array_buffer, 122536, 16384 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_14_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [32] },
      new Float32Array(weights_array_buffer, 138920, 128 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_16_filter_0 = builder.constant(
      { dataType: 'float32', shape: [128,32,1,1] },
      new Float32Array(weights_array_buffer, 139048, 16384 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_15_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [128] },
      new Float32Array(weights_array_buffer, 155432, 512 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_17_filter_0 = builder.constant(
      { dataType: 'float32', shape: [32,128,1,1] },
      new Float32Array(weights_array_buffer, 155944, 16384 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_16_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [32] },
      new Float32Array(weights_array_buffer, 172328, 128 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_18_filter_0 = builder.constant(
      { dataType: 'float32', shape: [128,32,1,1] },
      new Float32Array(weights_array_buffer, 172456, 16384 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_17_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [128] },
      new Float32Array(weights_array_buffer, 188840, 512 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const const_fold_opt__379 = builder.constant(
      { dataType: 'float32', shape: [128,1,5,5] },
      new Float32Array(weights_array_buffer, 189360, 12800 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_depthwise_conv2d_5_y_0 = builder.constant(
      { dataType: 'float32', shape: [128] },
      new Float32Array(weights_array_buffer, 202160, 512 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_19_filter_0 = builder.constant(
      { dataType: 'float32', shape: [32,128,1,1] },
      new Float32Array(weights_array_buffer, 202680, 16384 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_18_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [32] },
      new Float32Array(weights_array_buffer, 219064, 128 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_20_filter_0 = builder.constant(
      { dataType: 'float32', shape: [128,32,1,1] },
      new Float32Array(weights_array_buffer, 219192, 16384 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_19_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [128] },
      new Float32Array(weights_array_buffer, 235576, 512 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_21_filter_0 = builder.constant(
      { dataType: 'float32', shape: [32,128,1,1] },
      new Float32Array(weights_array_buffer, 236088, 16384 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_20_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [32] },
      new Float32Array(weights_array_buffer, 252472, 128 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_22_filter_0 = builder.constant(
      { dataType: 'float32', shape: [96,32,1,1] },
      new Float32Array(weights_array_buffer, 252600, 12288 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_21_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [96] },
      new Float32Array(weights_array_buffer, 264888, 384 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const const_fold_opt__395 = builder.constant(
      { dataType: 'float32', shape: [96,1,5,5] },
      new Float32Array(weights_array_buffer, 265280, 9600 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_depthwise_conv2d_6_y_0 = builder.constant(
      { dataType: 'float32', shape: [96] },
      new Float32Array(weights_array_buffer, 274880, 384 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_23_filter_0 = builder.constant(
      { dataType: 'float32', shape: [24,96,1,1] },
      new Float32Array(weights_array_buffer, 275272, 9216 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_22_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [24] },
      new Float32Array(weights_array_buffer, 284488, 96 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_24_filter_0 = builder.constant(
      { dataType: 'float32', shape: [96,24,1,1] },
      new Float32Array(weights_array_buffer, 284584, 9216 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_23_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [96] },
      new Float32Array(weights_array_buffer, 293800, 384 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_25_filter_0 = builder.constant(
      { dataType: 'float32', shape: [32,96,1,1] },
      new Float32Array(weights_array_buffer, 294184, 12288 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_24_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [32] },
      new Float32Array(weights_array_buffer, 306472, 128 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_26_filter_0 = builder.constant(
      { dataType: 'float32', shape: [96,32,1,1] },
      new Float32Array(weights_array_buffer, 306600, 12288 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_25_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [96] },
      new Float32Array(weights_array_buffer, 318888, 384 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const const_fold_opt__407 = builder.constant(
      { dataType: 'float32', shape: [96,1,5,5] },
      new Float32Array(weights_array_buffer, 319280, 9600 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_depthwise_conv2d_7_y_0 = builder.constant(
      { dataType: 'float32', shape: [96] },
      new Float32Array(weights_array_buffer, 328880, 384 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_27_filter_0 = builder.constant(
      { dataType: 'float32', shape: [24,96,1,1] },
      new Float32Array(weights_array_buffer, 329272, 9216 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_26_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [24] },
      new Float32Array(weights_array_buffer, 338488, 96 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_28_filter_0 = builder.constant(
      { dataType: 'float32', shape: [96,24,1,1] },
      new Float32Array(weights_array_buffer, 338584, 9216 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_27_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [96] },
      new Float32Array(weights_array_buffer, 347800, 384 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_29_filter_0 = builder.constant(
      { dataType: 'float32', shape: [32,96,1,1] },
      new Float32Array(weights_array_buffer, 348184, 12288 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_28_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [32] },
      new Float32Array(weights_array_buffer, 360472, 128 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_31_filter_0 = builder.constant(
      { dataType: 'float32', shape: [128,32,1,1] },
      new Float32Array(weights_array_buffer, 360600, 16384 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_30_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [128] },
      new Float32Array(weights_array_buffer, 376984, 512 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_30_filter_0 = builder.constant(
      { dataType: 'float32', shape: [128,32,1,1] },
      new Float32Array(weights_array_buffer, 377496, 16384 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_29_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [128] },
      new Float32Array(weights_array_buffer, 393880, 512 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_32_filter_0 = builder.constant(
      { dataType: 'float32', shape: [24,128,1,1] },
      new Float32Array(weights_array_buffer, 394392, 12288 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_31_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [24] },
      new Float32Array(weights_array_buffer, 406680, 96 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_33_filter_0 = builder.constant(
      { dataType: 'float32', shape: [24,24,1,1] },
      new Float32Array(weights_array_buffer, 406776, 2304 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_32_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [24] },
      new Float32Array(weights_array_buffer, 409080, 96 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_34_filter_0 = builder.constant(
      { dataType: 'float32', shape: [24,24,1,1] },
      new Float32Array(weights_array_buffer, 409176, 2304 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_33_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [24] },
      new Float32Array(weights_array_buffer, 411480, 96 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_35_filter_0 = builder.constant(
      { dataType: 'float32', shape: [24,24,1,1] },
      new Float32Array(weights_array_buffer, 411576, 2304 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_34_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [24] },
      new Float32Array(weights_array_buffer, 413880, 96 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const const_fold_opt__394 = builder.constant(
      { dataType: 'float32', shape: [24,1,3,3] },
      new Float32Array(weights_array_buffer, 413976, 864 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_depthwise_conv2d_8_y_0 = builder.constant(
      { dataType: 'float32', shape: [24] },
      new Float32Array(weights_array_buffer, 414840, 96 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_36_filter_0 = builder.constant(
      { dataType: 'float32', shape: [16,24,1,1] },
      new Float32Array(weights_array_buffer, 414936, 1536 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_35_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [16] },
      new Float32Array(weights_array_buffer, 416472, 64 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_37_filter_0 = builder.constant(
      { dataType: 'float32', shape: [16,16,1,1] },
      new Float32Array(weights_array_buffer, 416536, 1024 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_36_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [16] },
      new Float32Array(weights_array_buffer, 417560, 64 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_38_filter_0 = builder.constant(
      { dataType: 'float32', shape: [16,16,1,1] },
      new Float32Array(weights_array_buffer, 417624, 1024 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_37_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [16] },
      new Float32Array(weights_array_buffer, 418648, 64 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_39_filter_0 = builder.constant(
      { dataType: 'float32', shape: [16,16,1,1] },
      new Float32Array(weights_array_buffer, 418712, 1024 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_38_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [16] },
      new Float32Array(weights_array_buffer, 419736, 64 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const const_fold_opt__391 = builder.constant(
      { dataType: 'float32', shape: [16,1,3,3] },
      new Float32Array(weights_array_buffer, 419800, 576 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_depthwise_conv2d_9_y_0 = builder.constant(
      { dataType: 'float32', shape: [16] },
      new Float32Array(weights_array_buffer, 420376, 64 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_40_filter_0 = builder.constant(
      { dataType: 'float32', shape: [16,16,1,1] },
      new Float32Array(weights_array_buffer, 420440, 1024 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_39_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [16] },
      new Float32Array(weights_array_buffer, 421464, 64 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_41_filter_0 = builder.constant(
      { dataType: 'float32', shape: [16,16,1,1] },
      new Float32Array(weights_array_buffer, 421528, 1024 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_40_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [16] },
      new Float32Array(weights_array_buffer, 422552, 64 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_42_filter_0 = builder.constant(
      { dataType: 'float32', shape: [16,16,1,1] },
      new Float32Array(weights_array_buffer, 422616, 1024 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_41_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [16] },
      new Float32Array(weights_array_buffer, 423640, 64 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_43_filter_0 = builder.constant(
      { dataType: 'float32', shape: [16,16,1,1] },
      new Float32Array(weights_array_buffer, 423704, 1024 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_42_1_y_0 = builder.constant(
      { dataType: 'float32', shape: [16] },
      new Float32Array(weights_array_buffer, 424728, 64 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const const_fold_opt__389 = builder.constant(
      { dataType: 'float32', shape: [16,1,3,3] },
      new Float32Array(weights_array_buffer, 424792, 576 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_depthwise_conv2d_10_y_0 = builder.constant(
      { dataType: 'float32', shape: [16] },
      new Float32Array(weights_array_buffer, 425368, 64 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_conv2d_transpose_filter_0 = builder.constant(
      { dataType: 'float32', shape: [16,1,2,2] },
      new Float32Array(weights_array_buffer, 425432, 256 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const const_fold_opt__402 = builder.constant(
      { dataType: 'float32', shape: [1,1,1,1] },
      new Float32Array(weights_array_buffer, 425688, 4 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const new_shape__369 = builder.constant(
      { dataType: 'int64', shape: [4] },
      new BigInt64Array(weights_array_buffer.slice(425692, 425724))
    );
    

    // Create graph operators
        
    const var_conv2d__6_0 = builder.transpose(
      input
    );
    
    
    if (3 % 1 !== 0) throw new Error('The groups (1) must evenly divide the input channels (3)');
    if (3 !== 3 / 1) throw new Error('Filter input channels (3) must equal input channels (3) divided by groups (1)');
    
    const var_conv__158_0 = builder.conv2d(
      var_conv2d__6_0, var_conv2d_filter_0,
      {
        strides: [2, 2],
        padding: [0, 1, 0, 1],
        dilations: [1, 1],
        groups: 1,
        bias: var_conv2d_1_y_0
      }
    );
    
    const var_add_0 = builder.add(
      var_conv__158_0,
      const_fold_opt__408
    );
    
    const var_relu6_0 = builder.clamp(
      var_add_0,
      {
        minValue: var_292,
        maxValue: var_293
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
    
    
    if (16 % 1 !== 0) throw new Error('The groups (1) must evenly divide the input channels (16)');
    if (16 !== 16 / 1) throw new Error('Filter input channels (16) must equal input channels (16) divided by groups (1)');
    
    const var_conv__161_0 = builder.conv2d(
      var_mul_1_0, var_conv2d_2_filter_0,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_conv2d_1_1_y_0
      }
    );
    
    const var_re_lu_0 = builder.relu(
      var_conv__161_0
    );
    
    
    if (16 % 16 !== 0) throw new Error('The groups (16) must evenly divide the input channels (16)');
    if (1 !== 16 / 16) throw new Error('Filter input channels (1) must equal input channels (16) divided by groups (16)');
    
    const var_conv__162_0 = builder.conv2d(
      var_re_lu_0, const_fold_opt__377,
      {
        strides: [2, 2],
        padding: [0, 1, 0, 1],
        dilations: [1, 1],
        groups: 16,
        bias: var_depthwise_conv2d_y_0
      }
    );
    
    const var_re_lu_1_0 = builder.relu(
      var_conv__162_0
    );
    
    const var_global_average_pooling2d_0 = builder.averagePool2d(
      var_re_lu_1_0
    );
    
    
    if (16 % 1 !== 0) throw new Error('The groups (1) must evenly divide the input channels (16)');
    if (16 !== 16 / 1) throw new Error('Filter input channels (16) must equal input channels (16) divided by groups (1)');
    
    const var_conv__165_0 = builder.conv2d(
      var_global_average_pooling2d_0, var_conv2d_3_filter_0,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_conv2d_2_1_y_0
      }
    );
    
    const var_re_lu_2_0 = builder.relu(
      var_conv__165_0
    );
    
    
    if (8 % 1 !== 0) throw new Error('The groups (1) must evenly divide the input channels (8)');
    if (8 !== 8 / 1) throw new Error('Filter input channels (8) must equal input channels (8) divided by groups (1)');
    
    const var_conv__166_0 = builder.conv2d(
      var_re_lu_2_0, var_conv2d_4_filter_0,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_conv2d_3_1_y_0
      }
    );
    
    const var_activation_0 = builder.sigmoid(
      var_conv__166_0
    );
    
    const var_multiply_0 = builder.mul(
      var_re_lu_1_0,
      var_activation_0
    );
    
    
    if (16 % 1 !== 0) throw new Error('The groups (1) must evenly divide the input channels (16)');
    if (16 !== 16 / 1) throw new Error('Filter input channels (16) must equal input channels (16) divided by groups (1)');
    
    const var_conv__167_0 = builder.conv2d(
      var_multiply_0, var_conv2d_5_filter_0,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_conv2d_4_1_y_0
      }
    );
    
    
    if (16 % 1 !== 0) throw new Error('The groups (1) must evenly divide the input channels (16)');
    if (16 !== 16 / 1) throw new Error('Filter input channels (16) must equal input channels (16) divided by groups (1)');
    
    const var_conv__171_0 = builder.conv2d(
      var_conv__167_0, var_conv2d_6_filter_0,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_conv2d_5_1_y_0
      }
    );
    
    const var_re_lu_3_0 = builder.relu(
      var_conv__171_0
    );
    
    
    if (72 % 72 !== 0) throw new Error('The groups (72) must evenly divide the input channels (72)');
    if (1 !== 72 / 72) throw new Error('Filter input channels (1) must equal input channels (72) divided by groups (72)');
    
    const var_conv__172_0 = builder.conv2d(
      var_re_lu_3_0, const_fold_opt__387,
      {
        strides: [2, 2],
        padding: [0, 1, 0, 1],
        dilations: [1, 1],
        groups: 72,
        bias: var_depthwise_conv2d_1_y_0
      }
    );
    
    const var_re_lu_4_0 = builder.relu(
      var_conv__172_0
    );
    
    
    if (72 % 1 !== 0) throw new Error('The groups (1) must evenly divide the input channels (72)');
    if (72 !== 72 / 1) throw new Error('Filter input channels (72) must equal input channels (72) divided by groups (1)');
    
    const var_conv__173_0 = builder.conv2d(
      var_re_lu_4_0, var_conv2d_7_filter_0,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_conv2d_6_1_y_0
      }
    );
    
    
    if (24 % 1 !== 0) throw new Error('The groups (1) must evenly divide the input channels (24)');
    if (24 !== 24 / 1) throw new Error('Filter input channels (24) must equal input channels (24) divided by groups (1)');
    
    const var_conv__176_0 = builder.conv2d(
      var_conv__173_0, var_conv2d_8_filter_0,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_conv2d_7_1_y_0
      }
    );
    
    const var_re_lu_5_0 = builder.relu(
      var_conv__176_0
    );
    
    
    if (88 % 88 !== 0) throw new Error('The groups (88) must evenly divide the input channels (88)');
    if (1 !== 88 / 88) throw new Error('Filter input channels (1) must equal input channels (88) divided by groups (88)');
    
    const var_conv__177_0 = builder.conv2d(
      var_re_lu_5_0, const_fold_opt__385,
      {
        strides: [1, 1],
        padding: [1, 1, 1, 1],
        dilations: [1, 1],
        groups: 88,
        bias: var_depthwise_conv2d_2_y_0
      }
    );
    
    const var_re_lu_6_0 = builder.relu(
      var_conv__177_0
    );
    
    
    if (88 % 1 !== 0) throw new Error('The groups (1) must evenly divide the input channels (88)');
    if (88 !== 88 / 1) throw new Error('Filter input channels (88) must equal input channels (88) divided by groups (1)');
    
    const var_conv__178_0 = builder.conv2d(
      var_re_lu_6_0, var_conv2d_9_filter_0,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_conv2d_8_1_y_0
      }
    );
    
    const var_add__xeno_compat__1_0 = builder.add(
      var_conv__178_0,
      var_conv__173_0
    );
    
    
    if (24 % 1 !== 0) throw new Error('The groups (1) must evenly divide the input channels (24)');
    if (24 !== 24 / 1) throw new Error('Filter input channels (24) must equal input channels (24) divided by groups (1)');
    
    const var_conv__183_0 = builder.conv2d(
      var_add__xeno_compat__1_0, var_conv2d_10_filter_0,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_conv2d_9_1_y_0
      }
    );
    
    const var_add_1_0 = builder.add(
      var_conv__183_0,
      const_fold_opt__408
    );
    
    const var_relu6_1_0 = builder.clamp(
      var_add_1_0,
      {
        minValue: var_294,
        maxValue: var_295
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
    
    
    if (96 % 96 !== 0) throw new Error('The groups (96) must evenly divide the input channels (96)');
    if (1 !== 96 / 96) throw new Error('Filter input channels (1) must equal input channels (96) divided by groups (96)');
    
    const var_conv__186_0 = builder.conv2d(
      var_mul_3_0, const_fold_opt__383,
      {
        strides: [2, 2],
        padding: [1, 2, 1, 2],
        dilations: [1, 1],
        groups: 96,
        bias: var_depthwise_conv2d_3_y_0
      }
    );
    
    const var_add_2_0 = builder.add(
      var_conv__186_0,
      const_fold_opt__408
    );
    
    const var_relu6_2_0 = builder.clamp(
      var_add_2_0,
      {
        minValue: var_296,
        maxValue: var_297
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
    
    
    if (96 % 1 !== 0) throw new Error('The groups (1) must evenly divide the input channels (96)');
    if (96 !== 96 / 1) throw new Error('Filter input channels (96) must equal input channels (96) divided by groups (1)');
    
    const var_conv__189_0 = builder.conv2d(
      var_global_average_pooling2d_1_0, var_conv2d_11_filter_0,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_conv2d_10_1_y_0
      }
    );
    
    const var_re_lu_7_0 = builder.relu(
      var_conv__189_0
    );
    
    
    if (24 % 1 !== 0) throw new Error('The groups (1) must evenly divide the input channels (24)');
    if (24 !== 24 / 1) throw new Error('Filter input channels (24) must equal input channels (24) divided by groups (1)');
    
    const var_conv__190_0 = builder.conv2d(
      var_re_lu_7_0, var_conv2d_12_filter_0,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_conv2d_11_1_y_0
      }
    );
    
    const var_activation_1_0 = builder.sigmoid(
      var_conv__190_0
    );
    
    const var_multiply_1_0 = builder.mul(
      var_mul_5_0,
      var_activation_1_0
    );
    
    
    if (96 % 1 !== 0) throw new Error('The groups (1) must evenly divide the input channels (96)');
    if (96 !== 96 / 1) throw new Error('Filter input channels (96) must equal input channels (96) divided by groups (1)');
    
    const var_conv__191_0 = builder.conv2d(
      var_multiply_1_0, var_conv2d_13_filter_0,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_conv2d_12_1_y_0
      }
    );
    
    
    if (32 % 1 !== 0) throw new Error('The groups (1) must evenly divide the input channels (32)');
    if (32 !== 32 / 1) throw new Error('Filter input channels (32) must equal input channels (32) divided by groups (1)');
    
    const var_conv__194_0 = builder.conv2d(
      var_conv__191_0, var_conv2d_14_filter_0,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_conv2d_13_1_y_0
      }
    );
    
    const var_add_3_0 = builder.add(
      var_conv__194_0,
      const_fold_opt__408
    );
    
    const var_relu6_3_0 = builder.clamp(
      var_add_3_0,
      {
        minValue: var_298,
        maxValue: var_299
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
    
    
    if (128 % 128 !== 0) throw new Error('The groups (128) must evenly divide the input channels (128)');
    if (1 !== 128 / 128) throw new Error('Filter input channels (1) must equal input channels (128) divided by groups (128)');
    
    const var_conv__197_0 = builder.conv2d(
      var_mul_7_0, const_fold_opt__381,
      {
        strides: [1, 1],
        padding: [2, 2, 2, 2],
        dilations: [1, 1],
        groups: 128,
        bias: var_depthwise_conv2d_4_y_0
      }
    );
    
    const var_add_4_0 = builder.add(
      var_conv__197_0,
      const_fold_opt__408
    );
    
    const var_relu6_4_0 = builder.clamp(
      var_add_4_0,
      {
        minValue: var_300,
        maxValue: var_301
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
    
    
    if (128 % 1 !== 0) throw new Error('The groups (1) must evenly divide the input channels (128)');
    if (128 !== 128 / 1) throw new Error('Filter input channels (128) must equal input channels (128) divided by groups (1)');
    
    const var_conv__200_0 = builder.conv2d(
      var_global_average_pooling2d_2_0, var_conv2d_15_filter_0,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_conv2d_14_1_y_0
      }
    );
    
    const var_re_lu_8_0 = builder.relu(
      var_conv__200_0
    );
    
    
    if (32 % 1 !== 0) throw new Error('The groups (1) must evenly divide the input channels (32)');
    if (32 !== 32 / 1) throw new Error('Filter input channels (32) must equal input channels (32) divided by groups (1)');
    
    const var_conv__201_0 = builder.conv2d(
      var_re_lu_8_0, var_conv2d_16_filter_0,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_conv2d_15_1_y_0
      }
    );
    
    const var_activation_2_0 = builder.sigmoid(
      var_conv__201_0
    );
    
    const var_multiply_2_0 = builder.mul(
      var_mul_9_0,
      var_activation_2_0
    );
    
    
    if (128 % 1 !== 0) throw new Error('The groups (1) must evenly divide the input channels (128)');
    if (128 !== 128 / 1) throw new Error('Filter input channels (128) must equal input channels (128) divided by groups (1)');
    
    const var_conv__202_0 = builder.conv2d(
      var_multiply_2_0, var_conv2d_17_filter_0,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_conv2d_16_1_y_0
      }
    );
    
    const var_add_1__xeno_compat__1_0 = builder.add(
      var_conv__202_0,
      var_conv__191_0
    );
    
    
    if (32 % 1 !== 0) throw new Error('The groups (1) must evenly divide the input channels (32)');
    if (32 !== 32 / 1) throw new Error('Filter input channels (32) must equal input channels (32) divided by groups (1)');
    
    const var_conv__205_0 = builder.conv2d(
      var_add_1__xeno_compat__1_0, var_conv2d_18_filter_0,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_conv2d_17_1_y_0
      }
    );
    
    const var_add_5_0 = builder.add(
      var_conv__205_0,
      const_fold_opt__408
    );
    
    const var_relu6_5_0 = builder.clamp(
      var_add_5_0,
      {
        minValue: var_302,
        maxValue: var_303
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
    
    
    if (128 % 128 !== 0) throw new Error('The groups (128) must evenly divide the input channels (128)');
    if (1 !== 128 / 128) throw new Error('Filter input channels (1) must equal input channels (128) divided by groups (128)');
    
    const var_conv__208_0 = builder.conv2d(
      var_mul_11_0, const_fold_opt__379,
      {
        strides: [1, 1],
        padding: [2, 2, 2, 2],
        dilations: [1, 1],
        groups: 128,
        bias: var_depthwise_conv2d_5_y_0
      }
    );
    
    const var_add_6_0 = builder.add(
      var_conv__208_0,
      const_fold_opt__408
    );
    
    const var_relu6_6_0 = builder.clamp(
      var_add_6_0,
      {
        minValue: var_304,
        maxValue: var_305
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
    
    
    if (128 % 1 !== 0) throw new Error('The groups (1) must evenly divide the input channels (128)');
    if (128 !== 128 / 1) throw new Error('Filter input channels (128) must equal input channels (128) divided by groups (1)');
    
    const var_conv__211_0 = builder.conv2d(
      var_global_average_pooling2d_3_0, var_conv2d_19_filter_0,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_conv2d_18_1_y_0
      }
    );
    
    const var_re_lu_9_0 = builder.relu(
      var_conv__211_0
    );
    
    
    if (32 % 1 !== 0) throw new Error('The groups (1) must evenly divide the input channels (32)');
    if (32 !== 32 / 1) throw new Error('Filter input channels (32) must equal input channels (32) divided by groups (1)');
    
    const var_conv__212_0 = builder.conv2d(
      var_re_lu_9_0, var_conv2d_20_filter_0,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_conv2d_19_1_y_0
      }
    );
    
    const var_activation_3_0 = builder.sigmoid(
      var_conv__212_0
    );
    
    const var_multiply_3_0 = builder.mul(
      var_mul_13_0,
      var_activation_3_0
    );
    
    
    if (128 % 1 !== 0) throw new Error('The groups (1) must evenly divide the input channels (128)');
    if (128 !== 128 / 1) throw new Error('Filter input channels (128) must equal input channels (128) divided by groups (1)');
    
    const var_conv__213_0 = builder.conv2d(
      var_multiply_3_0, var_conv2d_21_filter_0,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_conv2d_20_1_y_0
      }
    );
    
    const var_add_2__xeno_compat__1_0 = builder.add(
      var_conv__213_0,
      var_add_1__xeno_compat__1_0
    );
    
    
    if (32 % 1 !== 0) throw new Error('The groups (1) must evenly divide the input channels (32)');
    if (32 !== 32 / 1) throw new Error('Filter input channels (32) must equal input channels (32) divided by groups (1)');
    
    const var_conv__216_0 = builder.conv2d(
      var_add_2__xeno_compat__1_0, var_conv2d_22_filter_0,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_conv2d_21_1_y_0
      }
    );
    
    const var_add_7_0 = builder.add(
      var_conv__216_0,
      const_fold_opt__408
    );
    
    const var_relu6_7_0 = builder.clamp(
      var_add_7_0,
      {
        minValue: var_306,
        maxValue: var_307
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
    
    
    if (96 % 96 !== 0) throw new Error('The groups (96) must evenly divide the input channels (96)');
    if (1 !== 96 / 96) throw new Error('Filter input channels (1) must equal input channels (96) divided by groups (96)');
    
    const var_conv__219_0 = builder.conv2d(
      var_mul_15_0, const_fold_opt__395,
      {
        strides: [1, 1],
        padding: [2, 2, 2, 2],
        dilations: [1, 1],
        groups: 96,
        bias: var_depthwise_conv2d_6_y_0
      }
    );
    
    const var_add_8_0 = builder.add(
      var_conv__219_0,
      const_fold_opt__408
    );
    
    const var_relu6_8_0 = builder.clamp(
      var_add_8_0,
      {
        minValue: var_308,
        maxValue: var_309
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
    
    
    if (96 % 1 !== 0) throw new Error('The groups (1) must evenly divide the input channels (96)');
    if (96 !== 96 / 1) throw new Error('Filter input channels (96) must equal input channels (96) divided by groups (1)');
    
    const var_conv__222_0 = builder.conv2d(
      var_global_average_pooling2d_4_0, var_conv2d_23_filter_0,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_conv2d_22_1_y_0
      }
    );
    
    const var_re_lu_10_0 = builder.relu(
      var_conv__222_0
    );
    
    
    if (24 % 1 !== 0) throw new Error('The groups (1) must evenly divide the input channels (24)');
    if (24 !== 24 / 1) throw new Error('Filter input channels (24) must equal input channels (24) divided by groups (1)');
    
    const var_conv__223_0 = builder.conv2d(
      var_re_lu_10_0, var_conv2d_24_filter_0,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_conv2d_23_1_y_0
      }
    );
    
    const var_activation_4_0 = builder.sigmoid(
      var_conv__223_0
    );
    
    const var_multiply_4_0 = builder.mul(
      var_mul_17_0,
      var_activation_4_0
    );
    
    
    if (96 % 1 !== 0) throw new Error('The groups (1) must evenly divide the input channels (96)');
    if (96 !== 96 / 1) throw new Error('Filter input channels (96) must equal input channels (96) divided by groups (1)');
    
    const var_conv__224_0 = builder.conv2d(
      var_multiply_4_0, var_conv2d_25_filter_0,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_conv2d_24_1_y_0
      }
    );
    
    const var_add_3__xeno_compat__1_0 = builder.add(
      var_conv__224_0,
      var_add_2__xeno_compat__1_0
    );
    
    
    if (32 % 1 !== 0) throw new Error('The groups (1) must evenly divide the input channels (32)');
    if (32 !== 32 / 1) throw new Error('Filter input channels (32) must equal input channels (32) divided by groups (1)');
    
    const var_conv__227_0 = builder.conv2d(
      var_add_3__xeno_compat__1_0, var_conv2d_26_filter_0,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_conv2d_25_1_y_0
      }
    );
    
    const var_add_9_0 = builder.add(
      var_conv__227_0,
      const_fold_opt__408
    );
    
    const var_relu6_9_0 = builder.clamp(
      var_add_9_0,
      {
        minValue: var_310,
        maxValue: var_311
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
    
    
    if (96 % 96 !== 0) throw new Error('The groups (96) must evenly divide the input channels (96)');
    if (1 !== 96 / 96) throw new Error('Filter input channels (1) must equal input channels (96) divided by groups (96)');
    
    const var_conv__230_0 = builder.conv2d(
      var_mul_19_0, const_fold_opt__407,
      {
        strides: [1, 1],
        padding: [2, 2, 2, 2],
        dilations: [1, 1],
        groups: 96,
        bias: var_depthwise_conv2d_7_y_0
      }
    );
    
    const var_add_10_0 = builder.add(
      var_conv__230_0,
      const_fold_opt__408
    );
    
    const var_relu6_10_0 = builder.clamp(
      var_add_10_0,
      {
        minValue: var_312,
        maxValue: var_313
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
    
    
    if (96 % 1 !== 0) throw new Error('The groups (1) must evenly divide the input channels (96)');
    if (96 !== 96 / 1) throw new Error('Filter input channels (96) must equal input channels (96) divided by groups (1)');
    
    const var_conv__233_0 = builder.conv2d(
      var_global_average_pooling2d_5_0, var_conv2d_27_filter_0,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_conv2d_26_1_y_0
      }
    );
    
    const var_re_lu_11_0 = builder.relu(
      var_conv__233_0
    );
    
    
    if (24 % 1 !== 0) throw new Error('The groups (1) must evenly divide the input channels (24)');
    if (24 !== 24 / 1) throw new Error('Filter input channels (24) must equal input channels (24) divided by groups (1)');
    
    const var_conv__234_0 = builder.conv2d(
      var_re_lu_11_0, var_conv2d_28_filter_0,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_conv2d_27_1_y_0
      }
    );
    
    const var_activation_5_0 = builder.sigmoid(
      var_conv__234_0
    );
    
    const var_multiply_5_0 = builder.mul(
      var_mul_21_0,
      var_activation_5_0
    );
    
    
    if (96 % 1 !== 0) throw new Error('The groups (1) must evenly divide the input channels (96)');
    if (96 !== 96 / 1) throw new Error('Filter input channels (96) must equal input channels (96) divided by groups (1)');
    
    const var_conv__235_0 = builder.conv2d(
      var_multiply_5_0, var_conv2d_29_filter_0,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_conv2d_28_1_y_0
      }
    );
    
    const var_add_4__xeno_compat__1_0 = builder.add(
      var_conv__235_0,
      var_add_3__xeno_compat__1_0
    );
    
    const var_global_average_pooling2d_6_0 = builder.averagePool2d(
      var_add_4__xeno_compat__1_0
    );
    
    
    if (32 % 1 !== 0) throw new Error('The groups (1) must evenly divide the input channels (32)');
    if (32 !== 32 / 1) throw new Error('Filter input channels (32) must equal input channels (32) divided by groups (1)');
    
    const var_conv__238_0 = builder.conv2d(
      var_global_average_pooling2d_6_0, var_conv2d_31_filter_0,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_conv2d_30_1_y_0
      }
    );
    
    const var_activation_6_0 = builder.sigmoid(
      var_conv__238_0
    );
    
    
    if (32 % 1 !== 0) throw new Error('The groups (1) must evenly divide the input channels (32)');
    if (32 !== 32 / 1) throw new Error('Filter input channels (32) must equal input channels (32) divided by groups (1)');
    
    const var_conv__239_0 = builder.conv2d(
      var_add_4__xeno_compat__1_0, var_conv2d_30_filter_0,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_conv2d_29_1_y_0
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
        mode: 'nearest-neighbor',
        scales: scales__125
      }
    );
    
    
    if (128 % 1 !== 0) throw new Error('The groups (1) must evenly divide the input channels (128)');
    if (128 !== 128 / 1) throw new Error('Filter input channels (128) must equal input channels (128) divided by groups (1)');
    
    const var_conv__240_0 = builder.conv2d(
      var_289, var_conv2d_32_filter_0,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_conv2d_31_1_y_0
      }
    );
    
    const var_add_5__xeno_compat__1_0 = builder.add(
      var_add__xeno_compat__1_0,
      var_conv__240_0
    );
    
    const var_global_average_pooling2d_7_0 = builder.averagePool2d(
      var_add_5__xeno_compat__1_0
    );
    
    
    if (24 % 1 !== 0) throw new Error('The groups (1) must evenly divide the input channels (24)');
    if (24 !== 24 / 1) throw new Error('Filter input channels (24) must equal input channels (24) divided by groups (1)');
    
    const var_conv__243_0 = builder.conv2d(
      var_global_average_pooling2d_7_0, var_conv2d_33_filter_0,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_conv2d_32_1_y_0
      }
    );
    
    const var_re_lu_13_0 = builder.relu(
      var_conv__243_0
    );
    
    
    if (24 % 1 !== 0) throw new Error('The groups (1) must evenly divide the input channels (24)');
    if (24 !== 24 / 1) throw new Error('Filter input channels (24) must equal input channels (24) divided by groups (1)');
    
    const var_conv__244_0 = builder.conv2d(
      var_re_lu_13_0, var_conv2d_34_filter_0,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_conv2d_33_1_y_0
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
    
    
    if (24 % 1 !== 0) throw new Error('The groups (1) must evenly divide the input channels (24)');
    if (24 !== 24 / 1) throw new Error('Filter input channels (24) must equal input channels (24) divided by groups (1)');
    
    const var_conv__245_0 = builder.conv2d(
      var_add_6__xeno_compat__1_0, var_conv2d_35_filter_0,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_conv2d_34_1_y_0
      }
    );
    
    const var_re_lu_14_0 = builder.relu(
      var_conv__245_0
    );
    
    
    if (24 % 24 !== 0) throw new Error('The groups (24) must evenly divide the input channels (24)');
    if (1 !== 24 / 24) throw new Error('Filter input channels (1) must equal input channels (24) divided by groups (24)');
    
    const var_conv__248_0 = builder.conv2d(
      var_re_lu_14_0, const_fold_opt__394,
      {
        strides: [1, 1],
        padding: [1, 1, 1, 1],
        dilations: [1, 1],
        groups: 24,
        bias: var_depthwise_conv2d_8_y_0
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
        mode: 'nearest-neighbor',
        scales: scales__125
      }
    );
    
    
    if (24 % 1 !== 0) throw new Error('The groups (1) must evenly divide the input channels (24)');
    if (24 !== 24 / 1) throw new Error('Filter input channels (24) must equal input channels (24) divided by groups (1)');
    
    const var_conv__249_0 = builder.conv2d(
      var_290, var_conv2d_36_filter_0,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_conv2d_35_1_y_0
      }
    );
    
    const var_add_8__xeno_compat__1_0 = builder.add(
      var_conv__167_0,
      var_conv__249_0
    );
    
    const var_global_average_pooling2d_8_0 = builder.averagePool2d(
      var_add_8__xeno_compat__1_0
    );
    
    
    if (16 % 1 !== 0) throw new Error('The groups (1) must evenly divide the input channels (16)');
    if (16 !== 16 / 1) throw new Error('Filter input channels (16) must equal input channels (16) divided by groups (1)');
    
    const var_conv__252_0 = builder.conv2d(
      var_global_average_pooling2d_8_0, var_conv2d_37_filter_0,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_conv2d_36_1_y_0
      }
    );
    
    const var_re_lu_16_0 = builder.relu(
      var_conv__252_0
    );
    
    
    if (16 % 1 !== 0) throw new Error('The groups (1) must evenly divide the input channels (16)');
    if (16 !== 16 / 1) throw new Error('Filter input channels (16) must equal input channels (16) divided by groups (1)');
    
    const var_conv__253_0 = builder.conv2d(
      var_re_lu_16_0, var_conv2d_38_filter_0,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_conv2d_37_1_y_0
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
    
    
    if (16 % 1 !== 0) throw new Error('The groups (1) must evenly divide the input channels (16)');
    if (16 !== 16 / 1) throw new Error('Filter input channels (16) must equal input channels (16) divided by groups (1)');
    
    const var_conv__254_0 = builder.conv2d(
      var_add_9__xeno_compat__1_0, var_conv2d_39_filter_0,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_conv2d_38_1_y_0
      }
    );
    
    const var_re_lu_17_0 = builder.relu(
      var_conv__254_0
    );
    
    
    if (16 % 16 !== 0) throw new Error('The groups (16) must evenly divide the input channels (16)');
    if (1 !== 16 / 16) throw new Error('Filter input channels (1) must equal input channels (16) divided by groups (16)');
    
    const var_conv__257_0 = builder.conv2d(
      var_re_lu_17_0, const_fold_opt__391,
      {
        strides: [1, 1],
        padding: [1, 1, 1, 1],
        dilations: [1, 1],
        groups: 16,
        bias: var_depthwise_conv2d_9_y_0
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
        mode: 'nearest-neighbor',
        scales: scales__125
      }
    );
    
    
    if (16 % 1 !== 0) throw new Error('The groups (1) must evenly divide the input channels (16)');
    if (16 !== 16 / 1) throw new Error('Filter input channels (16) must equal input channels (16) divided by groups (1)');
    
    const var_conv__258_0 = builder.conv2d(
      var_291, var_conv2d_40_filter_0,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_conv2d_39_1_y_0
      }
    );
    
    const var_add_11__xeno_compat__1_0 = builder.add(
      var_mul_1_0,
      var_conv__258_0
    );
    
    const var_global_average_pooling2d_9_0 = builder.averagePool2d(
      var_add_11__xeno_compat__1_0
    );
    
    
    if (16 % 1 !== 0) throw new Error('The groups (1) must evenly divide the input channels (16)');
    if (16 !== 16 / 1) throw new Error('Filter input channels (16) must equal input channels (16) divided by groups (1)');
    
    const var_conv__261_0 = builder.conv2d(
      var_global_average_pooling2d_9_0, var_conv2d_41_filter_0,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_conv2d_40_1_y_0
      }
    );
    
    const var_re_lu_19_0 = builder.relu(
      var_conv__261_0
    );
    
    
    if (16 % 1 !== 0) throw new Error('The groups (1) must evenly divide the input channels (16)');
    if (16 !== 16 / 1) throw new Error('Filter input channels (16) must equal input channels (16) divided by groups (1)');
    
    const var_conv__262_0 = builder.conv2d(
      var_re_lu_19_0, var_conv2d_42_filter_0,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_conv2d_41_1_y_0
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
    
    
    if (16 % 1 !== 0) throw new Error('The groups (1) must evenly divide the input channels (16)');
    if (16 !== 16 / 1) throw new Error('Filter input channels (16) must equal input channels (16) divided by groups (1)');
    
    const var_conv__263_0 = builder.conv2d(
      var_add_12__xeno_compat__1_0, var_conv2d_43_filter_0,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_conv2d_42_1_y_0
      }
    );
    
    const var_re_lu_20_0 = builder.relu(
      var_conv__263_0
    );
    
    
    if (16 % 16 !== 0) throw new Error('The groups (16) must evenly divide the input channels (16)');
    if (1 !== 16 / 16) throw new Error('Filter input channels (1) must equal input channels (16) divided by groups (16)');
    
    const var_conv__266_0 = builder.conv2d(
      var_re_lu_20_0, const_fold_opt__389,
      {
        strides: [1, 1],
        padding: [1, 1, 1, 1],
        dilations: [1, 1],
        groups: 16,
        bias: var_depthwise_conv2d_10_y_0
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
        const shape = Array.from(new BigInt64Array(weights_array_buffer, 425692, 32 / BigInt64Array.BYTES_PER_ELEMENT), Number);
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