// WebNN Code Generator (NHWC)

export class Squeezenet117Nhwc {

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
    
    const data = builder.transpose(
      builder.input('data', { dataType: 'float32', shape: [1,3,224,224] }),
      { permutation: [0, 2, 3, 1] }
    );

    this.inputTensors_['data'] = await this.context_.createTensor(
      { dataType: 'float32', shape: [1,3,224,224], writable: true }
    );

    // Initializers, create graph constant operands
    
    const squeezenet0_conv0_weight = builder.constant(
      { dataType: 'float32', shape: [64,3,3,3] },
      new Float32Array(weights_array_buffer, 0, 6912 / Float32Array.BYTES_PER_ELEMENT)
    );

    const squeezenet0_conv0_bias = builder.constant(
      { dataType: 'float32', shape: [64] },
      new Float32Array(weights_array_buffer, 6912, 256 / Float32Array.BYTES_PER_ELEMENT)
    );

    const squeezenet0_conv1_weight = builder.constant(
      { dataType: 'float32', shape: [16,64,1,1] },
      new Float32Array(weights_array_buffer, 7168, 4096 / Float32Array.BYTES_PER_ELEMENT)
    );

    const squeezenet0_conv1_bias = builder.constant(
      { dataType: 'float32', shape: [16] },
      new Float32Array([0.5199710130691528, 0.2506205141544342, -0.5917954444885254, 0.6274095177650452, 0.635230541229248, -0.36271199584007263, 0.1681842803955078, 0.3469047546386719, -0.2382478415966034, -0.5218156576156616, 0.14257831871509552, -0.7250925302505493, 0.4289284944534302, 0.1985836923122406, 0.25893792510032654, 0.5132268667221069])
    );

    const squeezenet0_conv2_weight = builder.constant(
      { dataType: 'float32', shape: [64,16,1,1] },
      new Float32Array(weights_array_buffer, 11328, 4096 / Float32Array.BYTES_PER_ELEMENT)
    );

    const squeezenet0_conv2_bias = builder.constant(
      { dataType: 'float32', shape: [64] },
      new Float32Array(weights_array_buffer, 15424, 256 / Float32Array.BYTES_PER_ELEMENT)
    );

    const squeezenet0_conv3_weight = builder.constant(
      { dataType: 'float32', shape: [64,16,3,3] },
      new Float32Array(weights_array_buffer, 15680, 36864 / Float32Array.BYTES_PER_ELEMENT)
    );

    const squeezenet0_conv3_bias = builder.constant(
      { dataType: 'float32', shape: [64] },
      new Float32Array(weights_array_buffer, 52544, 256 / Float32Array.BYTES_PER_ELEMENT)
    );

    const squeezenet0_conv4_weight = builder.constant(
      { dataType: 'float32', shape: [16,128,1,1] },
      new Float32Array(weights_array_buffer, 52800, 8192 / Float32Array.BYTES_PER_ELEMENT)
    );

    const squeezenet0_conv4_bias = builder.constant(
      { dataType: 'float32', shape: [16] },
      new Float32Array([0.17092423141002655, 0.797838032245636, 0.0872560366988182, 0.19740258157253265, 0.013273722492158413, 0.39042791724205017, 0.16029371321201324, 0.10641151666641235, 0.5180944800376892, 0.4875115752220154, -0.12850405275821686, 0.5590270757675171, 0.2691545784473419, 0.5232234001159668, 0.38500988483428955, -1.5481306314468384])
    );

    const squeezenet0_conv5_weight = builder.constant(
      { dataType: 'float32', shape: [64,16,1,1] },
      new Float32Array(weights_array_buffer, 61056, 4096 / Float32Array.BYTES_PER_ELEMENT)
    );

    const squeezenet0_conv5_bias = builder.constant(
      { dataType: 'float32', shape: [64] },
      new Float32Array(weights_array_buffer, 65152, 256 / Float32Array.BYTES_PER_ELEMENT)
    );

    const squeezenet0_conv6_weight = builder.constant(
      { dataType: 'float32', shape: [64,16,3,3] },
      new Float32Array(weights_array_buffer, 65408, 36864 / Float32Array.BYTES_PER_ELEMENT)
    );

    const squeezenet0_conv6_bias = builder.constant(
      { dataType: 'float32', shape: [64] },
      new Float32Array(weights_array_buffer, 102272, 256 / Float32Array.BYTES_PER_ELEMENT)
    );

    const squeezenet0_conv7_weight = builder.constant(
      { dataType: 'float32', shape: [32,128,1,1] },
      new Float32Array(weights_array_buffer, 102528, 16384 / Float32Array.BYTES_PER_ELEMENT)
    );

    const squeezenet0_conv7_bias = builder.constant(
      { dataType: 'float32', shape: [32] },
      new Float32Array(weights_array_buffer, 118912, 128 / Float32Array.BYTES_PER_ELEMENT)
    );

    const squeezenet0_conv8_weight = builder.constant(
      { dataType: 'float32', shape: [128,32,1,1] },
      new Float32Array(weights_array_buffer, 119040, 16384 / Float32Array.BYTES_PER_ELEMENT)
    );

    const squeezenet0_conv8_bias = builder.constant(
      { dataType: 'float32', shape: [128] },
      new Float32Array(weights_array_buffer, 135424, 512 / Float32Array.BYTES_PER_ELEMENT)
    );

    const squeezenet0_conv9_weight = builder.constant(
      { dataType: 'float32', shape: [128,32,3,3] },
      new Float32Array(weights_array_buffer, 135936, 147456 / Float32Array.BYTES_PER_ELEMENT)
    );

    const squeezenet0_conv9_bias = builder.constant(
      { dataType: 'float32', shape: [128] },
      new Float32Array(weights_array_buffer, 283392, 512 / Float32Array.BYTES_PER_ELEMENT)
    );

    const squeezenet0_conv10_weight = builder.constant(
      { dataType: 'float32', shape: [32,256,1,1] },
      new Float32Array(weights_array_buffer, 283904, 32768 / Float32Array.BYTES_PER_ELEMENT)
    );

    const squeezenet0_conv10_bias = builder.constant(
      { dataType: 'float32', shape: [32] },
      new Float32Array(weights_array_buffer, 316672, 128 / Float32Array.BYTES_PER_ELEMENT)
    );

    const squeezenet0_conv11_weight = builder.constant(
      { dataType: 'float32', shape: [128,32,1,1] },
      new Float32Array(weights_array_buffer, 316800, 16384 / Float32Array.BYTES_PER_ELEMENT)
    );

    const squeezenet0_conv11_bias = builder.constant(
      { dataType: 'float32', shape: [128] },
      new Float32Array(weights_array_buffer, 333184, 512 / Float32Array.BYTES_PER_ELEMENT)
    );

    const squeezenet0_conv12_weight = builder.constant(
      { dataType: 'float32', shape: [128,32,3,3] },
      new Float32Array(weights_array_buffer, 333696, 147456 / Float32Array.BYTES_PER_ELEMENT)
    );

    const squeezenet0_conv12_bias = builder.constant(
      { dataType: 'float32', shape: [128] },
      new Float32Array(weights_array_buffer, 481152, 512 / Float32Array.BYTES_PER_ELEMENT)
    );

    const squeezenet0_conv13_weight = builder.constant(
      { dataType: 'float32', shape: [48,256,1,1] },
      new Float32Array(weights_array_buffer, 481664, 49152 / Float32Array.BYTES_PER_ELEMENT)
    );

    const squeezenet0_conv13_bias = builder.constant(
      { dataType: 'float32', shape: [48] },
      new Float32Array(weights_array_buffer, 530816, 192 / Float32Array.BYTES_PER_ELEMENT)
    );

    const squeezenet0_conv14_weight = builder.constant(
      { dataType: 'float32', shape: [192,48,1,1] },
      new Float32Array(weights_array_buffer, 531008, 36864 / Float32Array.BYTES_PER_ELEMENT)
    );

    const squeezenet0_conv14_bias = builder.constant(
      { dataType: 'float32', shape: [192] },
      new Float32Array(weights_array_buffer, 567872, 768 / Float32Array.BYTES_PER_ELEMENT)
    );

    const squeezenet0_conv15_weight = builder.constant(
      { dataType: 'float32', shape: [192,48,3,3] },
      new Float32Array(weights_array_buffer, 568640, 331776 / Float32Array.BYTES_PER_ELEMENT)
    );

    const squeezenet0_conv15_bias = builder.constant(
      { dataType: 'float32', shape: [192] },
      new Float32Array(weights_array_buffer, 900416, 768 / Float32Array.BYTES_PER_ELEMENT)
    );

    const squeezenet0_conv16_weight = builder.constant(
      { dataType: 'float32', shape: [48,384,1,1] },
      new Float32Array(weights_array_buffer, 901184, 73728 / Float32Array.BYTES_PER_ELEMENT)
    );

    const squeezenet0_conv16_bias = builder.constant(
      { dataType: 'float32', shape: [48] },
      new Float32Array(weights_array_buffer, 974912, 192 / Float32Array.BYTES_PER_ELEMENT)
    );

    const squeezenet0_conv17_weight = builder.constant(
      { dataType: 'float32', shape: [192,48,1,1] },
      new Float32Array(weights_array_buffer, 975104, 36864 / Float32Array.BYTES_PER_ELEMENT)
    );

    const squeezenet0_conv17_bias = builder.constant(
      { dataType: 'float32', shape: [192] },
      new Float32Array(weights_array_buffer, 1011968, 768 / Float32Array.BYTES_PER_ELEMENT)
    );

    const squeezenet0_conv18_weight = builder.constant(
      { dataType: 'float32', shape: [192,48,3,3] },
      new Float32Array(weights_array_buffer, 1012736, 331776 / Float32Array.BYTES_PER_ELEMENT)
    );

    const squeezenet0_conv18_bias = builder.constant(
      { dataType: 'float32', shape: [192] },
      new Float32Array(weights_array_buffer, 1344512, 768 / Float32Array.BYTES_PER_ELEMENT)
    );

    const squeezenet0_conv19_weight = builder.constant(
      { dataType: 'float32', shape: [64,384,1,1] },
      new Float32Array(weights_array_buffer, 1345280, 98304 / Float32Array.BYTES_PER_ELEMENT)
    );

    const squeezenet0_conv19_bias = builder.constant(
      { dataType: 'float32', shape: [64] },
      new Float32Array(weights_array_buffer, 1443584, 256 / Float32Array.BYTES_PER_ELEMENT)
    );

    const squeezenet0_conv20_weight = builder.constant(
      { dataType: 'float32', shape: [256,64,1,1] },
      new Float32Array(weights_array_buffer, 1443840, 65536 / Float32Array.BYTES_PER_ELEMENT)
    );

    const squeezenet0_conv20_bias = builder.constant(
      { dataType: 'float32', shape: [256] },
      new Float32Array(weights_array_buffer, 1509376, 1024 / Float32Array.BYTES_PER_ELEMENT)
    );

    const squeezenet0_conv21_weight = builder.constant(
      { dataType: 'float32', shape: [256,64,3,3] },
      new Float32Array(weights_array_buffer, 1510400, 589824 / Float32Array.BYTES_PER_ELEMENT)
    );

    const squeezenet0_conv21_bias = builder.constant(
      { dataType: 'float32', shape: [256] },
      new Float32Array(weights_array_buffer, 2100224, 1024 / Float32Array.BYTES_PER_ELEMENT)
    );

    const squeezenet0_conv22_weight = builder.constant(
      { dataType: 'float32', shape: [64,512,1,1] },
      new Float32Array(weights_array_buffer, 2101248, 131072 / Float32Array.BYTES_PER_ELEMENT)
    );

    const squeezenet0_conv22_bias = builder.constant(
      { dataType: 'float32', shape: [64] },
      new Float32Array(weights_array_buffer, 2232320, 256 / Float32Array.BYTES_PER_ELEMENT)
    );

    const squeezenet0_conv23_weight = builder.constant(
      { dataType: 'float32', shape: [256,64,1,1] },
      new Float32Array(weights_array_buffer, 2232576, 65536 / Float32Array.BYTES_PER_ELEMENT)
    );

    const squeezenet0_conv23_bias = builder.constant(
      { dataType: 'float32', shape: [256] },
      new Float32Array(weights_array_buffer, 2298112, 1024 / Float32Array.BYTES_PER_ELEMENT)
    );

    const squeezenet0_conv24_weight = builder.constant(
      { dataType: 'float32', shape: [256,64,3,3] },
      new Float32Array(weights_array_buffer, 2299136, 589824 / Float32Array.BYTES_PER_ELEMENT)
    );

    const squeezenet0_conv24_bias = builder.constant(
      { dataType: 'float32', shape: [256] },
      new Float32Array(weights_array_buffer, 2888960, 1024 / Float32Array.BYTES_PER_ELEMENT)
    );

    const squeezenet0_conv25_weight = builder.constant(
      { dataType: 'float32', shape: [1000,512,1,1] },
      new Float32Array(weights_array_buffer, 2889984, 2048000 / Float32Array.BYTES_PER_ELEMENT)
    );

    const squeezenet0_conv25_bias = builder.constant(
      { dataType: 'float32', shape: [1000] },
      new Float32Array(weights_array_buffer, 4937984, 4000 / Float32Array.BYTES_PER_ELEMENT)
    );

    // Create graph operators
        
    const squeezenet0_conv0_fwd = builder.conv2d(
      data, squeezenet0_conv0_weight,
      {
        strides: [2, 2],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: squeezenet0_conv0_bias,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: 'squeezenet0_conv0_fwd'
      }
    );
    
    const squeezenet0_relu0_fwd = builder.relu(
      squeezenet0_conv0_fwd,
      { label: 'squeezenet0_relu0_fwd' }
    );
    
    const squeezenet0_pool0_fwd = builder.maxPool2d(
      squeezenet0_relu0_fwd,
      {
        windowDimensions: [3, 3],
        padding: [0, 0, 0, 0],
        strides: [2, 2],
        dilations: [1, 1],
        layout: 'nhwc',
        roundingType: 'floor',
        label: 'squeezenet0_pool0_fwd'
      }
    );
    
    const squeezenet0_conv1_fwd = builder.conv2d(
      squeezenet0_pool0_fwd, squeezenet0_conv1_weight,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: squeezenet0_conv1_bias,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: 'squeezenet0_conv1_fwd'
      }
    );
    
    const squeezenet0_relu1_fwd = builder.relu(
      squeezenet0_conv1_fwd,
      { label: 'squeezenet0_relu1_fwd' }
    );
    
    const squeezenet0_conv2_fwd = builder.conv2d(
      squeezenet0_relu1_fwd, squeezenet0_conv2_weight,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: squeezenet0_conv2_bias,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: 'squeezenet0_conv2_fwd'
      }
    );
    
    const squeezenet0_relu2_fwd = builder.relu(
      squeezenet0_conv2_fwd,
      { label: 'squeezenet0_relu2_fwd' }
    );
    
    const squeezenet0_conv3_fwd = builder.conv2d(
      squeezenet0_relu1_fwd, squeezenet0_conv3_weight,
      {
        strides: [1, 1],
        padding: [1, 1, 1, 1],
        dilations: [1, 1],
        groups: 1,
        bias: squeezenet0_conv3_bias,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: 'squeezenet0_conv3_fwd'
      }
    );
    
    const squeezenet0_relu3_fwd = builder.relu(
      squeezenet0_conv3_fwd,
      { label: 'squeezenet0_relu3_fwd' }
    );
    
    // Handle negative axis and ensure unsigned long for WebNN API
    let axis_squeezenet0_concat0 = 1;
    // If axis is negative, convert to positive based on input rank
    if (axis_squeezenet0_concat0 < 0) {
      // Use the first input's rank to resolve negative axis
      const firstInputRank = squeezenet0_relu2_fwd.shape.length;
      axis_squeezenet0_concat0 = firstInputRank + axis_squeezenet0_concat0;
    }
    // Ensure axis is a non-negative integer (unsigned long) as required by WebNN API
    axis_squeezenet0_concat0 = Math.max(0, Math.floor(axis_squeezenet0_concat0));

    const squeezenet0_concat0 = builder.concat(
      [squeezenet0_relu2_fwd, squeezenet0_relu3_fwd],
      axis_squeezenet0_concat0,
      { label: 'squeezenet0_concat0' }
    );
    
    const squeezenet0_conv4_fwd = builder.conv2d(
      squeezenet0_concat0, squeezenet0_conv4_weight,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: squeezenet0_conv4_bias,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: 'squeezenet0_conv4_fwd'
      }
    );
    
    const squeezenet0_relu4_fwd = builder.relu(
      squeezenet0_conv4_fwd,
      { label: 'squeezenet0_relu4_fwd' }
    );
    
    const squeezenet0_conv5_fwd = builder.conv2d(
      squeezenet0_relu4_fwd, squeezenet0_conv5_weight,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: squeezenet0_conv5_bias,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: 'squeezenet0_conv5_fwd'
      }
    );
    
    const squeezenet0_relu5_fwd = builder.relu(
      squeezenet0_conv5_fwd,
      { label: 'squeezenet0_relu5_fwd' }
    );
    
    const squeezenet0_conv6_fwd = builder.conv2d(
      squeezenet0_relu4_fwd, squeezenet0_conv6_weight,
      {
        strides: [1, 1],
        padding: [1, 1, 1, 1],
        dilations: [1, 1],
        groups: 1,
        bias: squeezenet0_conv6_bias,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: 'squeezenet0_conv6_fwd'
      }
    );
    
    const squeezenet0_relu6_fwd = builder.relu(
      squeezenet0_conv6_fwd,
      { label: 'squeezenet0_relu6_fwd' }
    );
    
    // Handle negative axis and ensure unsigned long for WebNN API
    let axis_squeezenet0_concat1 = 1;
    // If axis is negative, convert to positive based on input rank
    if (axis_squeezenet0_concat1 < 0) {
      // Use the first input's rank to resolve negative axis
      const firstInputRank = squeezenet0_relu5_fwd.shape.length;
      axis_squeezenet0_concat1 = firstInputRank + axis_squeezenet0_concat1;
    }
    // Ensure axis is a non-negative integer (unsigned long) as required by WebNN API
    axis_squeezenet0_concat1 = Math.max(0, Math.floor(axis_squeezenet0_concat1));

    const squeezenet0_concat1 = builder.concat(
      [squeezenet0_relu5_fwd, squeezenet0_relu6_fwd],
      axis_squeezenet0_concat1,
      { label: 'squeezenet0_concat1' }
    );
    
    const squeezenet0_pool1_fwd = builder.maxPool2d(
      squeezenet0_concat1,
      {
        windowDimensions: [3, 3],
        padding: [0, 0, 0, 0],
        strides: [2, 2],
        dilations: [1, 1],
        layout: 'nhwc',
        roundingType: 'floor',
        label: 'squeezenet0_pool1_fwd'
      }
    );
    
    const squeezenet0_conv7_fwd = builder.conv2d(
      squeezenet0_pool1_fwd, squeezenet0_conv7_weight,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: squeezenet0_conv7_bias,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: 'squeezenet0_conv7_fwd'
      }
    );
    
    const squeezenet0_relu7_fwd = builder.relu(
      squeezenet0_conv7_fwd,
      { label: 'squeezenet0_relu7_fwd' }
    );
    
    const squeezenet0_conv8_fwd = builder.conv2d(
      squeezenet0_relu7_fwd, squeezenet0_conv8_weight,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: squeezenet0_conv8_bias,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: 'squeezenet0_conv8_fwd'
      }
    );
    
    const squeezenet0_relu8_fwd = builder.relu(
      squeezenet0_conv8_fwd,
      { label: 'squeezenet0_relu8_fwd' }
    );
    
    const squeezenet0_conv9_fwd = builder.conv2d(
      squeezenet0_relu7_fwd, squeezenet0_conv9_weight,
      {
        strides: [1, 1],
        padding: [1, 1, 1, 1],
        dilations: [1, 1],
        groups: 1,
        bias: squeezenet0_conv9_bias,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: 'squeezenet0_conv9_fwd'
      }
    );
    
    const squeezenet0_relu9_fwd = builder.relu(
      squeezenet0_conv9_fwd,
      { label: 'squeezenet0_relu9_fwd' }
    );
    
    // Handle negative axis and ensure unsigned long for WebNN API
    let axis_squeezenet0_concat2 = 1;
    // If axis is negative, convert to positive based on input rank
    if (axis_squeezenet0_concat2 < 0) {
      // Use the first input's rank to resolve negative axis
      const firstInputRank = squeezenet0_relu8_fwd.shape.length;
      axis_squeezenet0_concat2 = firstInputRank + axis_squeezenet0_concat2;
    }
    // Ensure axis is a non-negative integer (unsigned long) as required by WebNN API
    axis_squeezenet0_concat2 = Math.max(0, Math.floor(axis_squeezenet0_concat2));

    const squeezenet0_concat2 = builder.concat(
      [squeezenet0_relu8_fwd, squeezenet0_relu9_fwd],
      axis_squeezenet0_concat2,
      { label: 'squeezenet0_concat2' }
    );
    
    const squeezenet0_conv10_fwd = builder.conv2d(
      squeezenet0_concat2, squeezenet0_conv10_weight,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: squeezenet0_conv10_bias,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: 'squeezenet0_conv10_fwd'
      }
    );
    
    const squeezenet0_relu10_fwd = builder.relu(
      squeezenet0_conv10_fwd,
      { label: 'squeezenet0_relu10_fwd' }
    );
    
    const squeezenet0_conv11_fwd = builder.conv2d(
      squeezenet0_relu10_fwd, squeezenet0_conv11_weight,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: squeezenet0_conv11_bias,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: 'squeezenet0_conv11_fwd'
      }
    );
    
    const squeezenet0_relu11_fwd = builder.relu(
      squeezenet0_conv11_fwd,
      { label: 'squeezenet0_relu11_fwd' }
    );
    
    const squeezenet0_conv12_fwd = builder.conv2d(
      squeezenet0_relu10_fwd, squeezenet0_conv12_weight,
      {
        strides: [1, 1],
        padding: [1, 1, 1, 1],
        dilations: [1, 1],
        groups: 1,
        bias: squeezenet0_conv12_bias,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: 'squeezenet0_conv12_fwd'
      }
    );
    
    const squeezenet0_relu12_fwd = builder.relu(
      squeezenet0_conv12_fwd,
      { label: 'squeezenet0_relu12_fwd' }
    );
    
    // Handle negative axis and ensure unsigned long for WebNN API
    let axis_squeezenet0_concat3 = 1;
    // If axis is negative, convert to positive based on input rank
    if (axis_squeezenet0_concat3 < 0) {
      // Use the first input's rank to resolve negative axis
      const firstInputRank = squeezenet0_relu11_fwd.shape.length;
      axis_squeezenet0_concat3 = firstInputRank + axis_squeezenet0_concat3;
    }
    // Ensure axis is a non-negative integer (unsigned long) as required by WebNN API
    axis_squeezenet0_concat3 = Math.max(0, Math.floor(axis_squeezenet0_concat3));

    const squeezenet0_concat3 = builder.concat(
      [squeezenet0_relu11_fwd, squeezenet0_relu12_fwd],
      axis_squeezenet0_concat3,
      { label: 'squeezenet0_concat3' }
    );
    
    const squeezenet0_pool2_fwd = builder.maxPool2d(
      squeezenet0_concat3,
      {
        windowDimensions: [3, 3],
        padding: [0, 0, 0, 0],
        strides: [2, 2],
        dilations: [1, 1],
        layout: 'nhwc',
        roundingType: 'floor',
        label: 'squeezenet0_pool2_fwd'
      }
    );
    
    const squeezenet0_conv13_fwd = builder.conv2d(
      squeezenet0_pool2_fwd, squeezenet0_conv13_weight,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: squeezenet0_conv13_bias,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: 'squeezenet0_conv13_fwd'
      }
    );
    
    const squeezenet0_relu13_fwd = builder.relu(
      squeezenet0_conv13_fwd,
      { label: 'squeezenet0_relu13_fwd' }
    );
    
    const squeezenet0_conv14_fwd = builder.conv2d(
      squeezenet0_relu13_fwd, squeezenet0_conv14_weight,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: squeezenet0_conv14_bias,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: 'squeezenet0_conv14_fwd'
      }
    );
    
    const squeezenet0_relu14_fwd = builder.relu(
      squeezenet0_conv14_fwd,
      { label: 'squeezenet0_relu14_fwd' }
    );
    
    const squeezenet0_conv15_fwd = builder.conv2d(
      squeezenet0_relu13_fwd, squeezenet0_conv15_weight,
      {
        strides: [1, 1],
        padding: [1, 1, 1, 1],
        dilations: [1, 1],
        groups: 1,
        bias: squeezenet0_conv15_bias,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: 'squeezenet0_conv15_fwd'
      }
    );
    
    const squeezenet0_relu15_fwd = builder.relu(
      squeezenet0_conv15_fwd,
      { label: 'squeezenet0_relu15_fwd' }
    );
    
    // Handle negative axis and ensure unsigned long for WebNN API
    let axis_squeezenet0_concat4 = 1;
    // If axis is negative, convert to positive based on input rank
    if (axis_squeezenet0_concat4 < 0) {
      // Use the first input's rank to resolve negative axis
      const firstInputRank = squeezenet0_relu14_fwd.shape.length;
      axis_squeezenet0_concat4 = firstInputRank + axis_squeezenet0_concat4;
    }
    // Ensure axis is a non-negative integer (unsigned long) as required by WebNN API
    axis_squeezenet0_concat4 = Math.max(0, Math.floor(axis_squeezenet0_concat4));

    const squeezenet0_concat4 = builder.concat(
      [squeezenet0_relu14_fwd, squeezenet0_relu15_fwd],
      axis_squeezenet0_concat4,
      { label: 'squeezenet0_concat4' }
    );
    
    const squeezenet0_conv16_fwd = builder.conv2d(
      squeezenet0_concat4, squeezenet0_conv16_weight,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: squeezenet0_conv16_bias,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: 'squeezenet0_conv16_fwd'
      }
    );
    
    const squeezenet0_relu16_fwd = builder.relu(
      squeezenet0_conv16_fwd,
      { label: 'squeezenet0_relu16_fwd' }
    );
    
    const squeezenet0_conv17_fwd = builder.conv2d(
      squeezenet0_relu16_fwd, squeezenet0_conv17_weight,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: squeezenet0_conv17_bias,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: 'squeezenet0_conv17_fwd'
      }
    );
    
    const squeezenet0_relu17_fwd = builder.relu(
      squeezenet0_conv17_fwd,
      { label: 'squeezenet0_relu17_fwd' }
    );
    
    const squeezenet0_conv18_fwd = builder.conv2d(
      squeezenet0_relu16_fwd, squeezenet0_conv18_weight,
      {
        strides: [1, 1],
        padding: [1, 1, 1, 1],
        dilations: [1, 1],
        groups: 1,
        bias: squeezenet0_conv18_bias,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: 'squeezenet0_conv18_fwd'
      }
    );
    
    const squeezenet0_relu18_fwd = builder.relu(
      squeezenet0_conv18_fwd,
      { label: 'squeezenet0_relu18_fwd' }
    );
    
    // Handle negative axis and ensure unsigned long for WebNN API
    let axis_squeezenet0_concat5 = 1;
    // If axis is negative, convert to positive based on input rank
    if (axis_squeezenet0_concat5 < 0) {
      // Use the first input's rank to resolve negative axis
      const firstInputRank = squeezenet0_relu17_fwd.shape.length;
      axis_squeezenet0_concat5 = firstInputRank + axis_squeezenet0_concat5;
    }
    // Ensure axis is a non-negative integer (unsigned long) as required by WebNN API
    axis_squeezenet0_concat5 = Math.max(0, Math.floor(axis_squeezenet0_concat5));

    const squeezenet0_concat5 = builder.concat(
      [squeezenet0_relu17_fwd, squeezenet0_relu18_fwd],
      axis_squeezenet0_concat5,
      { label: 'squeezenet0_concat5' }
    );
    
    const squeezenet0_conv19_fwd = builder.conv2d(
      squeezenet0_concat5, squeezenet0_conv19_weight,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: squeezenet0_conv19_bias,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: 'squeezenet0_conv19_fwd'
      }
    );
    
    const squeezenet0_relu19_fwd = builder.relu(
      squeezenet0_conv19_fwd,
      { label: 'squeezenet0_relu19_fwd' }
    );
    
    const squeezenet0_conv20_fwd = builder.conv2d(
      squeezenet0_relu19_fwd, squeezenet0_conv20_weight,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: squeezenet0_conv20_bias,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: 'squeezenet0_conv20_fwd'
      }
    );
    
    const squeezenet0_relu20_fwd = builder.relu(
      squeezenet0_conv20_fwd,
      { label: 'squeezenet0_relu20_fwd' }
    );
    
    const squeezenet0_conv21_fwd = builder.conv2d(
      squeezenet0_relu19_fwd, squeezenet0_conv21_weight,
      {
        strides: [1, 1],
        padding: [1, 1, 1, 1],
        dilations: [1, 1],
        groups: 1,
        bias: squeezenet0_conv21_bias,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: 'squeezenet0_conv21_fwd'
      }
    );
    
    const squeezenet0_relu21_fwd = builder.relu(
      squeezenet0_conv21_fwd,
      { label: 'squeezenet0_relu21_fwd' }
    );
    
    // Handle negative axis and ensure unsigned long for WebNN API
    let axis_squeezenet0_concat6 = 1;
    // If axis is negative, convert to positive based on input rank
    if (axis_squeezenet0_concat6 < 0) {
      // Use the first input's rank to resolve negative axis
      const firstInputRank = squeezenet0_relu20_fwd.shape.length;
      axis_squeezenet0_concat6 = firstInputRank + axis_squeezenet0_concat6;
    }
    // Ensure axis is a non-negative integer (unsigned long) as required by WebNN API
    axis_squeezenet0_concat6 = Math.max(0, Math.floor(axis_squeezenet0_concat6));

    const squeezenet0_concat6 = builder.concat(
      [squeezenet0_relu20_fwd, squeezenet0_relu21_fwd],
      axis_squeezenet0_concat6,
      { label: 'squeezenet0_concat6' }
    );
    
    const squeezenet0_conv22_fwd = builder.conv2d(
      squeezenet0_concat6, squeezenet0_conv22_weight,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: squeezenet0_conv22_bias,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: 'squeezenet0_conv22_fwd'
      }
    );
    
    const squeezenet0_relu22_fwd = builder.relu(
      squeezenet0_conv22_fwd,
      { label: 'squeezenet0_relu22_fwd' }
    );
    
    const squeezenet0_conv23_fwd = builder.conv2d(
      squeezenet0_relu22_fwd, squeezenet0_conv23_weight,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: squeezenet0_conv23_bias,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: 'squeezenet0_conv23_fwd'
      }
    );
    
    const squeezenet0_relu23_fwd = builder.relu(
      squeezenet0_conv23_fwd,
      { label: 'squeezenet0_relu23_fwd' }
    );
    
    const squeezenet0_conv24_fwd = builder.conv2d(
      squeezenet0_relu22_fwd, squeezenet0_conv24_weight,
      {
        strides: [1, 1],
        padding: [1, 1, 1, 1],
        dilations: [1, 1],
        groups: 1,
        bias: squeezenet0_conv24_bias,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: 'squeezenet0_conv24_fwd'
      }
    );
    
    const squeezenet0_relu24_fwd = builder.relu(
      squeezenet0_conv24_fwd,
      { label: 'squeezenet0_relu24_fwd' }
    );
    
    // Handle negative axis and ensure unsigned long for WebNN API
    let axis_squeezenet0_concat7 = 1;
    // If axis is negative, convert to positive based on input rank
    if (axis_squeezenet0_concat7 < 0) {
      // Use the first input's rank to resolve negative axis
      const firstInputRank = squeezenet0_relu23_fwd.shape.length;
      axis_squeezenet0_concat7 = firstInputRank + axis_squeezenet0_concat7;
    }
    // Ensure axis is a non-negative integer (unsigned long) as required by WebNN API
    axis_squeezenet0_concat7 = Math.max(0, Math.floor(axis_squeezenet0_concat7));

    const squeezenet0_concat7 = builder.concat(
      [squeezenet0_relu23_fwd, squeezenet0_relu24_fwd],
      axis_squeezenet0_concat7,
      { label: 'squeezenet0_concat7' }
    );
    
      const squeezenet0_dropout0_fwd = builder.identity(
        squeezenet0_concat7,
        { label: 'squeezenet0_dropout0_fwd' }
      );
    
    const squeezenet0_conv25_fwd = builder.conv2d(
      squeezenet0_dropout0_fwd, squeezenet0_conv25_weight,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: squeezenet0_conv25_bias,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: 'squeezenet0_conv25_fwd'
      }
    );
    
    const squeezenet0_relu25_fwd = builder.relu(
      squeezenet0_conv25_fwd,
      { label: 'squeezenet0_relu25_fwd' }
    );
    
    const squeezenet0_pool3_fwd = builder.averagePool2d(
      squeezenet0_relu25_fwd,
      {
        windowDimensions: [13, 13],
        strides: [13, 13],
        dilations: [1, 1],
        roundingType: 'floor',
        layout: 'nhwc',
        padding: [0, 0, 0, 0],
        label: 'squeezenet0_pool3_fwd' 
      }
    );
    
    const squeezenet0_flatten0_reshape0 = builder.reshape(
      squeezenet0_pool3_fwd,
      (() => {
      // If newShape's size is 0, then set outputShape to an empty list for a scalar
      const initialShape = [0,-1];
      if (initialShape.length === 0) {
        return [];
      }
      
      let shape = [...initialShape];
      
      // Handle 0 dimensions (copy from input shape)
      const inputShape = squeezenet0_pool3_fwd.shape;
      for (let i = 0; i < shape.length; i++) {
        if (shape[i] === 0 && i < inputShape.length) {
          shape[i] = inputShape[i];
        }
      }
      
      // Calculate the concrete size for value -1
      if (shape.includes(-1)) {
        const count = shape.filter(v => v === -1).length;
        if (count !== 1) {
          throw new TypeError('Only one -1 is allowed in reshape shape');
        }
        
        // Calculate inputElementCount (product of all items in input's shape)
        const inputElementCount = inputShape.reduce((a, b) => a * b, 1);
        
        // Calculate known (product of all values in shape except -1)
        const known = shape.reduce((a, b) => b === -1 ? a : a * b, 1);
        
        if (known === 0) {
          throw new TypeError('Product of shape dimensions contains 0');
        }
        
        const idx = shape.indexOf(-1);
        const inferredDim = Math.floor(inputElementCount / known);
        
        // Check if the inferred dimension results in the same number of elements
        if (inferredDim * known !== inputElementCount) {
          throw new TypeError('Total size of input tensor is not divisible by product of specified dimensions');
        }
        
        shape[idx] = inferredDim;
      }
      
      // Validate the shape: ensure all values are valid unsigned long integers
      const outputShape = shape.map(dim => {
        if (isNaN(dim) || !isFinite(dim) || dim < 0) {
          throw new TypeError('Shape dimension must be a non-negative integer');
        }
        return Math.floor(Number(dim));
      });
      
      // Check if product of newShape equals inputElementCount
      const inputElementCount = inputShape.reduce((a, b) => a * b, 1);
      const outputElementCount = outputShape.reduce((a, b) => a * b, 1);
      
      if (outputElementCount !== inputElementCount) {
        throw new TypeError('Product of output shape dimensions must equal the product of input shape dimensions');
      }
      
      return outputShape;
    })(),
      { label: 'squeezenet0_flatten0_reshape0' }
    );

    // Build graph with all outputs
    
    this.graph_ = await builder.build({ 'squeezenet0_flatten0_reshape0': squeezenet0_flatten0_reshape0 });

    // Create output tensors
    
    this.outputTensors_['squeezenet0_flatten0_reshape0'] = await this.context_.createTensor(
      { dataType: 'float32', shape: [1,1000], readable: true }
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