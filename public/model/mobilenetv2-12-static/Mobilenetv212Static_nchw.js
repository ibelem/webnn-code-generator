// WebNN Code Generator (NCHW)

export class Mobilenetv212StaticNchw {

  constructor() {
    this.context_ = null;
    this.graph_ = null;
    this.inputTensors_ = {};
    this.outputTensors_ = {};
  }

  async build(options) {
    // Load weights ArrayBuffer from .bin file
    async function loadWeightsArrayBuffer() {
      const binFile = 'weights_nchw.bin';
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
    
    const input = builder.input('input', { dataType: 'float32', shape: [1,3,224,224] });
    this.inputTensors_['input'] = await this.context_.createTensor(
      { dataType: 'float32', shape: [1,3,224,224], writable: true }
    );

    // Create graph constant operands
    
    const var_475 = builder.constant(
      { dataType: 'float32', shape: [32,3,3,3] },
      new Float32Array(weights_array_buffer, 0, 3456 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_476 = builder.constant(
      { dataType: 'float32', shape: [32] },
      new Float32Array(weights_array_buffer, 3456, 128 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const clip_3min = builder.constant(
      { dataType: 'float32', shape: [1] },
      new Float32Array([0])
    );

    const clip_19max = builder.constant(
      { dataType: 'float32', shape: [1] },
      new Float32Array([6])
    );

    const var_478 = builder.constant(
      { dataType: 'float32', shape: [32,1,3,3] },
      new Float32Array(weights_array_buffer, 3600, 1152 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_479 = builder.constant(
      { dataType: 'float32', shape: [32] },
      new Float32Array(weights_array_buffer, 4752, 128 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_481 = builder.constant(
      { dataType: 'float32', shape: [16,32,1,1] },
      new Float32Array(weights_array_buffer, 4896, 2048 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_482 = builder.constant(
      { dataType: 'float32', shape: [16] },
      new Float32Array(weights_array_buffer, 6944, 64 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_484 = builder.constant(
      { dataType: 'float32', shape: [96,16,1,1] },
      new Float32Array(weights_array_buffer, 7008, 6144 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_485 = builder.constant(
      { dataType: 'float32', shape: [96] },
      new Float32Array(weights_array_buffer, 13152, 384 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_487 = builder.constant(
      { dataType: 'float32', shape: [96,1,3,3] },
      new Float32Array(weights_array_buffer, 13552, 3456 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_488 = builder.constant(
      { dataType: 'float32', shape: [96] },
      new Float32Array(weights_array_buffer, 17008, 384 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_490 = builder.constant(
      { dataType: 'float32', shape: [24,96,1,1] },
      new Float32Array(weights_array_buffer, 17408, 9216 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_491 = builder.constant(
      { dataType: 'float32', shape: [24] },
      new Float32Array(weights_array_buffer, 26624, 96 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_493 = builder.constant(
      { dataType: 'float32', shape: [144,24,1,1] },
      new Float32Array(weights_array_buffer, 26720, 13824 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_494 = builder.constant(
      { dataType: 'float32', shape: [144] },
      new Float32Array(weights_array_buffer, 40544, 576 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_496 = builder.constant(
      { dataType: 'float32', shape: [144,1,3,3] },
      new Float32Array(weights_array_buffer, 41136, 5184 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_497 = builder.constant(
      { dataType: 'float32', shape: [144] },
      new Float32Array(weights_array_buffer, 46320, 576 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_499 = builder.constant(
      { dataType: 'float32', shape: [24,144,1,1] },
      new Float32Array(weights_array_buffer, 46912, 13824 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_500 = builder.constant(
      { dataType: 'float32', shape: [24] },
      new Float32Array(weights_array_buffer, 60736, 96 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_502 = builder.constant(
      { dataType: 'float32', shape: [144,24,1,1] },
      new Float32Array(weights_array_buffer, 60832, 13824 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_503 = builder.constant(
      { dataType: 'float32', shape: [144] },
      new Float32Array(weights_array_buffer, 74656, 576 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_505 = builder.constant(
      { dataType: 'float32', shape: [144,1,3,3] },
      new Float32Array(weights_array_buffer, 75248, 5184 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_506 = builder.constant(
      { dataType: 'float32', shape: [144] },
      new Float32Array(weights_array_buffer, 80432, 576 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_508 = builder.constant(
      { dataType: 'float32', shape: [32,144,1,1] },
      new Float32Array(weights_array_buffer, 81024, 18432 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_509 = builder.constant(
      { dataType: 'float32', shape: [32] },
      new Float32Array(weights_array_buffer, 99456, 128 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_511 = builder.constant(
      { dataType: 'float32', shape: [192,32,1,1] },
      new Float32Array(weights_array_buffer, 99584, 24576 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_512 = builder.constant(
      { dataType: 'float32', shape: [192] },
      new Float32Array(weights_array_buffer, 124160, 768 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_514 = builder.constant(
      { dataType: 'float32', shape: [192,1,3,3] },
      new Float32Array(weights_array_buffer, 124944, 6912 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_515 = builder.constant(
      { dataType: 'float32', shape: [192] },
      new Float32Array(weights_array_buffer, 131856, 768 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_517 = builder.constant(
      { dataType: 'float32', shape: [32,192,1,1] },
      new Float32Array(weights_array_buffer, 132640, 24576 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_518 = builder.constant(
      { dataType: 'float32', shape: [32] },
      new Float32Array(weights_array_buffer, 157216, 128 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_520 = builder.constant(
      { dataType: 'float32', shape: [192,32,1,1] },
      new Float32Array(weights_array_buffer, 157344, 24576 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_521 = builder.constant(
      { dataType: 'float32', shape: [192] },
      new Float32Array(weights_array_buffer, 181920, 768 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_523 = builder.constant(
      { dataType: 'float32', shape: [192,1,3,3] },
      new Float32Array(weights_array_buffer, 182704, 6912 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_524 = builder.constant(
      { dataType: 'float32', shape: [192] },
      new Float32Array(weights_array_buffer, 189616, 768 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_526 = builder.constant(
      { dataType: 'float32', shape: [32,192,1,1] },
      new Float32Array(weights_array_buffer, 190400, 24576 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_527 = builder.constant(
      { dataType: 'float32', shape: [32] },
      new Float32Array(weights_array_buffer, 214976, 128 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_529 = builder.constant(
      { dataType: 'float32', shape: [192,32,1,1] },
      new Float32Array(weights_array_buffer, 215104, 24576 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_530 = builder.constant(
      { dataType: 'float32', shape: [192] },
      new Float32Array(weights_array_buffer, 239680, 768 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_532 = builder.constant(
      { dataType: 'float32', shape: [192,1,3,3] },
      new Float32Array(weights_array_buffer, 240464, 6912 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_533 = builder.constant(
      { dataType: 'float32', shape: [192] },
      new Float32Array(weights_array_buffer, 247376, 768 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_535 = builder.constant(
      { dataType: 'float32', shape: [64,192,1,1] },
      new Float32Array(weights_array_buffer, 248160, 49152 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_536 = builder.constant(
      { dataType: 'float32', shape: [64] },
      new Float32Array(weights_array_buffer, 297312, 256 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_538 = builder.constant(
      { dataType: 'float32', shape: [384,64,1,1] },
      new Float32Array(weights_array_buffer, 297568, 98304 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_539 = builder.constant(
      { dataType: 'float32', shape: [384] },
      new Float32Array(weights_array_buffer, 395872, 1536 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_541 = builder.constant(
      { dataType: 'float32', shape: [384,1,3,3] },
      new Float32Array(weights_array_buffer, 397424, 13824 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_542 = builder.constant(
      { dataType: 'float32', shape: [384] },
      new Float32Array(weights_array_buffer, 411248, 1536 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_544 = builder.constant(
      { dataType: 'float32', shape: [64,384,1,1] },
      new Float32Array(weights_array_buffer, 412800, 98304 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_545 = builder.constant(
      { dataType: 'float32', shape: [64] },
      new Float32Array(weights_array_buffer, 511104, 256 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_547 = builder.constant(
      { dataType: 'float32', shape: [384,64,1,1] },
      new Float32Array(weights_array_buffer, 511360, 98304 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_548 = builder.constant(
      { dataType: 'float32', shape: [384] },
      new Float32Array(weights_array_buffer, 609664, 1536 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_550 = builder.constant(
      { dataType: 'float32', shape: [384,1,3,3] },
      new Float32Array(weights_array_buffer, 611216, 13824 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_551 = builder.constant(
      { dataType: 'float32', shape: [384] },
      new Float32Array(weights_array_buffer, 625040, 1536 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_553 = builder.constant(
      { dataType: 'float32', shape: [64,384,1,1] },
      new Float32Array(weights_array_buffer, 626592, 98304 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_554 = builder.constant(
      { dataType: 'float32', shape: [64] },
      new Float32Array(weights_array_buffer, 724896, 256 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_556 = builder.constant(
      { dataType: 'float32', shape: [384,64,1,1] },
      new Float32Array(weights_array_buffer, 725152, 98304 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_557 = builder.constant(
      { dataType: 'float32', shape: [384] },
      new Float32Array(weights_array_buffer, 823456, 1536 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_559 = builder.constant(
      { dataType: 'float32', shape: [384,1,3,3] },
      new Float32Array(weights_array_buffer, 825008, 13824 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_560 = builder.constant(
      { dataType: 'float32', shape: [384] },
      new Float32Array(weights_array_buffer, 838832, 1536 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_562 = builder.constant(
      { dataType: 'float32', shape: [64,384,1,1] },
      new Float32Array(weights_array_buffer, 840384, 98304 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_563 = builder.constant(
      { dataType: 'float32', shape: [64] },
      new Float32Array(weights_array_buffer, 938688, 256 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_565 = builder.constant(
      { dataType: 'float32', shape: [384,64,1,1] },
      new Float32Array(weights_array_buffer, 938944, 98304 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_566 = builder.constant(
      { dataType: 'float32', shape: [384] },
      new Float32Array(weights_array_buffer, 1037248, 1536 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_568 = builder.constant(
      { dataType: 'float32', shape: [384,1,3,3] },
      new Float32Array(weights_array_buffer, 1038800, 13824 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_569 = builder.constant(
      { dataType: 'float32', shape: [384] },
      new Float32Array(weights_array_buffer, 1052624, 1536 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_571 = builder.constant(
      { dataType: 'float32', shape: [96,384,1,1] },
      new Float32Array(weights_array_buffer, 1054176, 147456 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_572 = builder.constant(
      { dataType: 'float32', shape: [96] },
      new Float32Array(weights_array_buffer, 1201632, 384 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_574 = builder.constant(
      { dataType: 'float32', shape: [576,96,1,1] },
      new Float32Array(weights_array_buffer, 1202016, 221184 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_575 = builder.constant(
      { dataType: 'float32', shape: [576] },
      new Float32Array(weights_array_buffer, 1423200, 2304 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_577 = builder.constant(
      { dataType: 'float32', shape: [576,1,3,3] },
      new Float32Array(weights_array_buffer, 1425520, 20736 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_578 = builder.constant(
      { dataType: 'float32', shape: [576] },
      new Float32Array(weights_array_buffer, 1446256, 2304 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_580 = builder.constant(
      { dataType: 'float32', shape: [96,576,1,1] },
      new Float32Array(weights_array_buffer, 1448576, 221184 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_581 = builder.constant(
      { dataType: 'float32', shape: [96] },
      new Float32Array(weights_array_buffer, 1669760, 384 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_583 = builder.constant(
      { dataType: 'float32', shape: [576,96,1,1] },
      new Float32Array(weights_array_buffer, 1670144, 221184 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_584 = builder.constant(
      { dataType: 'float32', shape: [576] },
      new Float32Array(weights_array_buffer, 1891328, 2304 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_586 = builder.constant(
      { dataType: 'float32', shape: [576,1,3,3] },
      new Float32Array(weights_array_buffer, 1893648, 20736 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_587 = builder.constant(
      { dataType: 'float32', shape: [576] },
      new Float32Array(weights_array_buffer, 1914384, 2304 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_589 = builder.constant(
      { dataType: 'float32', shape: [96,576,1,1] },
      new Float32Array(weights_array_buffer, 1916704, 221184 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_590 = builder.constant(
      { dataType: 'float32', shape: [96] },
      new Float32Array(weights_array_buffer, 2137888, 384 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_592 = builder.constant(
      { dataType: 'float32', shape: [576,96,1,1] },
      new Float32Array(weights_array_buffer, 2138272, 221184 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_593 = builder.constant(
      { dataType: 'float32', shape: [576] },
      new Float32Array(weights_array_buffer, 2359456, 2304 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_595 = builder.constant(
      { dataType: 'float32', shape: [576,1,3,3] },
      new Float32Array(weights_array_buffer, 2361776, 20736 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_596 = builder.constant(
      { dataType: 'float32', shape: [576] },
      new Float32Array(weights_array_buffer, 2382512, 2304 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_598 = builder.constant(
      { dataType: 'float32', shape: [160,576,1,1] },
      new Float32Array(weights_array_buffer, 2384832, 368640 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_599 = builder.constant(
      { dataType: 'float32', shape: [160] },
      new Float32Array(weights_array_buffer, 2753472, 640 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_601 = builder.constant(
      { dataType: 'float32', shape: [960,160,1,1] },
      new Float32Array(weights_array_buffer, 2754112, 614400 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_602 = builder.constant(
      { dataType: 'float32', shape: [960] },
      new Float32Array(weights_array_buffer, 3368512, 3840 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_604 = builder.constant(
      { dataType: 'float32', shape: [960,1,3,3] },
      new Float32Array(weights_array_buffer, 3372368, 34560 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_605 = builder.constant(
      { dataType: 'float32', shape: [960] },
      new Float32Array(weights_array_buffer, 3406928, 3840 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_607 = builder.constant(
      { dataType: 'float32', shape: [160,960,1,1] },
      new Float32Array(weights_array_buffer, 3410784, 614400 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_608 = builder.constant(
      { dataType: 'float32', shape: [160] },
      new Float32Array(weights_array_buffer, 4025184, 640 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_610 = builder.constant(
      { dataType: 'float32', shape: [960,160,1,1] },
      new Float32Array(weights_array_buffer, 4025824, 614400 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_611 = builder.constant(
      { dataType: 'float32', shape: [960] },
      new Float32Array(weights_array_buffer, 4640224, 3840 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_613 = builder.constant(
      { dataType: 'float32', shape: [960,1,3,3] },
      new Float32Array(weights_array_buffer, 4644080, 34560 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_614 = builder.constant(
      { dataType: 'float32', shape: [960] },
      new Float32Array(weights_array_buffer, 4678640, 3840 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_616 = builder.constant(
      { dataType: 'float32', shape: [160,960,1,1] },
      new Float32Array(weights_array_buffer, 4682496, 614400 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_617 = builder.constant(
      { dataType: 'float32', shape: [160] },
      new Float32Array(weights_array_buffer, 5296896, 640 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_619 = builder.constant(
      { dataType: 'float32', shape: [960,160,1,1] },
      new Float32Array(weights_array_buffer, 5297536, 614400 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_620 = builder.constant(
      { dataType: 'float32', shape: [960] },
      new Float32Array(weights_array_buffer, 5911936, 3840 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_622 = builder.constant(
      { dataType: 'float32', shape: [960,1,3,3] },
      new Float32Array(weights_array_buffer, 5915792, 34560 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_623 = builder.constant(
      { dataType: 'float32', shape: [960] },
      new Float32Array(weights_array_buffer, 5950352, 3840 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_625 = builder.constant(
      { dataType: 'float32', shape: [320,960,1,1] },
      new Float32Array(weights_array_buffer, 5954208, 1228800 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_626 = builder.constant(
      { dataType: 'float32', shape: [320] },
      new Float32Array(weights_array_buffer, 7183008, 1280 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_628 = builder.constant(
      { dataType: 'float32', shape: [1280,320,1,1] },
      new Float32Array(weights_array_buffer, 7184288, 1638400 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_629 = builder.constant(
      { dataType: 'float32', shape: [1280] },
      new Float32Array(weights_array_buffer, 8822688, 5120 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_471 = builder.constant(
      { dataType: 'int64', shape: [2] },
      new BigInt64Array(weights_array_buffer, 8827824, 16 / BigInt64Array.BYTES_PER_ELEMENT)
    );
    
    const var_classifier_1_weight = builder.constant(
      { dataType: 'float32', shape: [1000,1280] },
      new Float32Array(weights_array_buffer, 8827840, 5120000 / Float32Array.BYTES_PER_ELEMENT)
    );
    
    const var_classifier_1_bias = builder.constant(
      { dataType: 'float32', shape: [1000] },
      new Float32Array(weights_array_buffer, 13947840, 4000 / Float32Array.BYTES_PER_ELEMENT)
    );
    

    // Create graph operators
        
    const var_474 = builder.conv2d(
      input, var_475,
      {
        strides: [2, 2],
        padding: [1, 1, 1, 1],
        dilations: [1, 1],
        groups: 1,
        bias: var_476,
        filterLayout: 'oihw',
        inputLayout: 'nchw',
        label: 'Conv_0'
      }
    );
  
    
    const var_317 = builder.clamp(
      var_474,
      {
        minValue: 0,
        maxValue: 6
      }
    );
    
    const var_477 = builder.conv2d(
      var_317, var_478,
      {
        strides: [1, 1],
        padding: [1, 1, 1, 1],
        dilations: [1, 1],
        groups: 32,
        bias: var_479,
        filterLayout: 'oihw',
        inputLayout: 'nchw',
        label: 'Conv_2'
      }
    );
  
    
    const var_320 = builder.clamp(
      var_477,
      {
        minValue: 0,
        maxValue: 6
      }
    );
    
    const var_480 = builder.conv2d(
      var_320, var_481,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_482,
        filterLayout: 'oihw',
        inputLayout: 'nchw',
        label: 'Conv_4'
      }
    );
  
    
    const var_483 = builder.conv2d(
      var_480, var_484,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_485,
        filterLayout: 'oihw',
        inputLayout: 'nchw',
        label: 'Conv_5'
      }
    );
  
    
    const var_325 = builder.clamp(
      var_483,
      {
        minValue: 0,
        maxValue: 6
      }
    );
    
    const var_486 = builder.conv2d(
      var_325, var_487,
      {
        strides: [2, 2],
        padding: [1, 1, 1, 1],
        dilations: [1, 1],
        groups: 96,
        bias: var_488,
        filterLayout: 'oihw',
        inputLayout: 'nchw',
        label: 'Conv_7'
      }
    );
  
    
    const var_328 = builder.clamp(
      var_486,
      {
        minValue: 0,
        maxValue: 6
      }
    );
    
    const var_489 = builder.conv2d(
      var_328, var_490,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_491,
        filterLayout: 'oihw',
        inputLayout: 'nchw',
        label: 'Conv_9'
      }
    );
  
    
    const var_492 = builder.conv2d(
      var_489, var_493,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_494,
        filterLayout: 'oihw',
        inputLayout: 'nchw',
        label: 'Conv_10'
      }
    );
  
    
    const var_333 = builder.clamp(
      var_492,
      {
        minValue: 0,
        maxValue: 6
      }
    );
    
    const var_495 = builder.conv2d(
      var_333, var_496,
      {
        strides: [1, 1],
        padding: [1, 1, 1, 1],
        dilations: [1, 1],
        groups: 144,
        bias: var_497,
        filterLayout: 'oihw',
        inputLayout: 'nchw',
        label: 'Conv_12'
      }
    );
  
    
    const var_336 = builder.clamp(
      var_495,
      {
        minValue: 0,
        maxValue: 6
      }
    );
    
    const var_498 = builder.conv2d(
      var_336, var_499,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_500,
        filterLayout: 'oihw',
        inputLayout: 'nchw',
        label: 'Conv_14'
      }
    );
  
    
    const var_339 = builder.add(
      var_489,
      var_498
    );
    
    const var_501 = builder.conv2d(
      var_339, var_502,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_503,
        filterLayout: 'oihw',
        inputLayout: 'nchw',
        label: 'Conv_16'
      }
    );
  
    
    const var_342 = builder.clamp(
      var_501,
      {
        minValue: 0,
        maxValue: 6
      }
    );
    
    const var_504 = builder.conv2d(
      var_342, var_505,
      {
        strides: [2, 2],
        padding: [1, 1, 1, 1],
        dilations: [1, 1],
        groups: 144,
        bias: var_506,
        filterLayout: 'oihw',
        inputLayout: 'nchw',
        label: 'Conv_18'
      }
    );
  
    
    const var_345 = builder.clamp(
      var_504,
      {
        minValue: 0,
        maxValue: 6
      }
    );
    
    const var_507 = builder.conv2d(
      var_345, var_508,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_509,
        filterLayout: 'oihw',
        inputLayout: 'nchw',
        label: 'Conv_20'
      }
    );
  
    
    const var_510 = builder.conv2d(
      var_507, var_511,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_512,
        filterLayout: 'oihw',
        inputLayout: 'nchw',
        label: 'Conv_21'
      }
    );
  
    
    const var_350 = builder.clamp(
      var_510,
      {
        minValue: 0,
        maxValue: 6
      }
    );
    
    const var_513 = builder.conv2d(
      var_350, var_514,
      {
        strides: [1, 1],
        padding: [1, 1, 1, 1],
        dilations: [1, 1],
        groups: 192,
        bias: var_515,
        filterLayout: 'oihw',
        inputLayout: 'nchw',
        label: 'Conv_23'
      }
    );
  
    
    const var_353 = builder.clamp(
      var_513,
      {
        minValue: 0,
        maxValue: 6
      }
    );
    
    const var_516 = builder.conv2d(
      var_353, var_517,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_518,
        filterLayout: 'oihw',
        inputLayout: 'nchw',
        label: 'Conv_25'
      }
    );
  
    
    const var_356 = builder.add(
      var_507,
      var_516
    );
    
    const var_519 = builder.conv2d(
      var_356, var_520,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_521,
        filterLayout: 'oihw',
        inputLayout: 'nchw',
        label: 'Conv_27'
      }
    );
  
    
    const var_359 = builder.clamp(
      var_519,
      {
        minValue: 0,
        maxValue: 6
      }
    );
    
    const var_522 = builder.conv2d(
      var_359, var_523,
      {
        strides: [1, 1],
        padding: [1, 1, 1, 1],
        dilations: [1, 1],
        groups: 192,
        bias: var_524,
        filterLayout: 'oihw',
        inputLayout: 'nchw',
        label: 'Conv_29'
      }
    );
  
    
    const var_362 = builder.clamp(
      var_522,
      {
        minValue: 0,
        maxValue: 6
      }
    );
    
    const var_525 = builder.conv2d(
      var_362, var_526,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_527,
        filterLayout: 'oihw',
        inputLayout: 'nchw',
        label: 'Conv_31'
      }
    );
  
    
    const var_365 = builder.add(
      var_356,
      var_525
    );
    
    const var_528 = builder.conv2d(
      var_365, var_529,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_530,
        filterLayout: 'oihw',
        inputLayout: 'nchw',
        label: 'Conv_33'
      }
    );
  
    
    const var_368 = builder.clamp(
      var_528,
      {
        minValue: 0,
        maxValue: 6
      }
    );
    
    const var_531 = builder.conv2d(
      var_368, var_532,
      {
        strides: [2, 2],
        padding: [1, 1, 1, 1],
        dilations: [1, 1],
        groups: 192,
        bias: var_533,
        filterLayout: 'oihw',
        inputLayout: 'nchw',
        label: 'Conv_35'
      }
    );
  
    
    const var_371 = builder.clamp(
      var_531,
      {
        minValue: 0,
        maxValue: 6
      }
    );
    
    const var_534 = builder.conv2d(
      var_371, var_535,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_536,
        filterLayout: 'oihw',
        inputLayout: 'nchw',
        label: 'Conv_37'
      }
    );
  
    
    const var_537 = builder.conv2d(
      var_534, var_538,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_539,
        filterLayout: 'oihw',
        inputLayout: 'nchw',
        label: 'Conv_38'
      }
    );
  
    
    const var_376 = builder.clamp(
      var_537,
      {
        minValue: 0,
        maxValue: 6
      }
    );
    
    const var_540 = builder.conv2d(
      var_376, var_541,
      {
        strides: [1, 1],
        padding: [1, 1, 1, 1],
        dilations: [1, 1],
        groups: 384,
        bias: var_542,
        filterLayout: 'oihw',
        inputLayout: 'nchw',
        label: 'Conv_40'
      }
    );
  
    
    const var_379 = builder.clamp(
      var_540,
      {
        minValue: 0,
        maxValue: 6
      }
    );
    
    const var_543 = builder.conv2d(
      var_379, var_544,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_545,
        filterLayout: 'oihw',
        inputLayout: 'nchw',
        label: 'Conv_42'
      }
    );
  
    
    const var_382 = builder.add(
      var_534,
      var_543
    );
    
    const var_546 = builder.conv2d(
      var_382, var_547,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_548,
        filterLayout: 'oihw',
        inputLayout: 'nchw',
        label: 'Conv_44'
      }
    );
  
    
    const var_385 = builder.clamp(
      var_546,
      {
        minValue: 0,
        maxValue: 6
      }
    );
    
    const var_549 = builder.conv2d(
      var_385, var_550,
      {
        strides: [1, 1],
        padding: [1, 1, 1, 1],
        dilations: [1, 1],
        groups: 384,
        bias: var_551,
        filterLayout: 'oihw',
        inputLayout: 'nchw',
        label: 'Conv_46'
      }
    );
  
    
    const var_388 = builder.clamp(
      var_549,
      {
        minValue: 0,
        maxValue: 6
      }
    );
    
    const var_552 = builder.conv2d(
      var_388, var_553,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_554,
        filterLayout: 'oihw',
        inputLayout: 'nchw',
        label: 'Conv_48'
      }
    );
  
    
    const var_391 = builder.add(
      var_382,
      var_552
    );
    
    const var_555 = builder.conv2d(
      var_391, var_556,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_557,
        filterLayout: 'oihw',
        inputLayout: 'nchw',
        label: 'Conv_50'
      }
    );
  
    
    const var_394 = builder.clamp(
      var_555,
      {
        minValue: 0,
        maxValue: 6
      }
    );
    
    const var_558 = builder.conv2d(
      var_394, var_559,
      {
        strides: [1, 1],
        padding: [1, 1, 1, 1],
        dilations: [1, 1],
        groups: 384,
        bias: var_560,
        filterLayout: 'oihw',
        inputLayout: 'nchw',
        label: 'Conv_52'
      }
    );
  
    
    const var_397 = builder.clamp(
      var_558,
      {
        minValue: 0,
        maxValue: 6
      }
    );
    
    const var_561 = builder.conv2d(
      var_397, var_562,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_563,
        filterLayout: 'oihw',
        inputLayout: 'nchw',
        label: 'Conv_54'
      }
    );
  
    
    const var_400 = builder.add(
      var_391,
      var_561
    );
    
    const var_564 = builder.conv2d(
      var_400, var_565,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_566,
        filterLayout: 'oihw',
        inputLayout: 'nchw',
        label: 'Conv_56'
      }
    );
  
    
    const var_403 = builder.clamp(
      var_564,
      {
        minValue: 0,
        maxValue: 6
      }
    );
    
    const var_567 = builder.conv2d(
      var_403, var_568,
      {
        strides: [1, 1],
        padding: [1, 1, 1, 1],
        dilations: [1, 1],
        groups: 384,
        bias: var_569,
        filterLayout: 'oihw',
        inputLayout: 'nchw',
        label: 'Conv_58'
      }
    );
  
    
    const var_406 = builder.clamp(
      var_567,
      {
        minValue: 0,
        maxValue: 6
      }
    );
    
    const var_570 = builder.conv2d(
      var_406, var_571,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_572,
        filterLayout: 'oihw',
        inputLayout: 'nchw',
        label: 'Conv_60'
      }
    );
  
    
    const var_573 = builder.conv2d(
      var_570, var_574,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_575,
        filterLayout: 'oihw',
        inputLayout: 'nchw',
        label: 'Conv_61'
      }
    );
  
    
    const var_411 = builder.clamp(
      var_573,
      {
        minValue: 0,
        maxValue: 6
      }
    );
    
    const var_576 = builder.conv2d(
      var_411, var_577,
      {
        strides: [1, 1],
        padding: [1, 1, 1, 1],
        dilations: [1, 1],
        groups: 576,
        bias: var_578,
        filterLayout: 'oihw',
        inputLayout: 'nchw',
        label: 'Conv_63'
      }
    );
  
    
    const var_414 = builder.clamp(
      var_576,
      {
        minValue: 0,
        maxValue: 6
      }
    );
    
    const var_579 = builder.conv2d(
      var_414, var_580,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_581,
        filterLayout: 'oihw',
        inputLayout: 'nchw',
        label: 'Conv_65'
      }
    );
  
    
    const var_417 = builder.add(
      var_570,
      var_579
    );
    
    const var_582 = builder.conv2d(
      var_417, var_583,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_584,
        filterLayout: 'oihw',
        inputLayout: 'nchw',
        label: 'Conv_67'
      }
    );
  
    
    const var_420 = builder.clamp(
      var_582,
      {
        minValue: 0,
        maxValue: 6
      }
    );
    
    const var_585 = builder.conv2d(
      var_420, var_586,
      {
        strides: [1, 1],
        padding: [1, 1, 1, 1],
        dilations: [1, 1],
        groups: 576,
        bias: var_587,
        filterLayout: 'oihw',
        inputLayout: 'nchw',
        label: 'Conv_69'
      }
    );
  
    
    const var_423 = builder.clamp(
      var_585,
      {
        minValue: 0,
        maxValue: 6
      }
    );
    
    const var_588 = builder.conv2d(
      var_423, var_589,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_590,
        filterLayout: 'oihw',
        inputLayout: 'nchw',
        label: 'Conv_71'
      }
    );
  
    
    const var_426 = builder.add(
      var_417,
      var_588
    );
    
    const var_591 = builder.conv2d(
      var_426, var_592,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_593,
        filterLayout: 'oihw',
        inputLayout: 'nchw',
        label: 'Conv_73'
      }
    );
  
    
    const var_429 = builder.clamp(
      var_591,
      {
        minValue: 0,
        maxValue: 6
      }
    );
    
    const var_594 = builder.conv2d(
      var_429, var_595,
      {
        strides: [2, 2],
        padding: [1, 1, 1, 1],
        dilations: [1, 1],
        groups: 576,
        bias: var_596,
        filterLayout: 'oihw',
        inputLayout: 'nchw',
        label: 'Conv_75'
      }
    );
  
    
    const var_432 = builder.clamp(
      var_594,
      {
        minValue: 0,
        maxValue: 6
      }
    );
    
    const var_597 = builder.conv2d(
      var_432, var_598,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_599,
        filterLayout: 'oihw',
        inputLayout: 'nchw',
        label: 'Conv_77'
      }
    );
  
    
    const var_600 = builder.conv2d(
      var_597, var_601,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_602,
        filterLayout: 'oihw',
        inputLayout: 'nchw',
        label: 'Conv_78'
      }
    );
  
    
    const var_437 = builder.clamp(
      var_600,
      {
        minValue: 0,
        maxValue: 6
      }
    );
    
    const var_603 = builder.conv2d(
      var_437, var_604,
      {
        strides: [1, 1],
        padding: [1, 1, 1, 1],
        dilations: [1, 1],
        groups: 960,
        bias: var_605,
        filterLayout: 'oihw',
        inputLayout: 'nchw',
        label: 'Conv_80'
      }
    );
  
    
    const var_440 = builder.clamp(
      var_603,
      {
        minValue: 0,
        maxValue: 6
      }
    );
    
    const var_606 = builder.conv2d(
      var_440, var_607,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_608,
        filterLayout: 'oihw',
        inputLayout: 'nchw',
        label: 'Conv_82'
      }
    );
  
    
    const var_443 = builder.add(
      var_597,
      var_606
    );
    
    const var_609 = builder.conv2d(
      var_443, var_610,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_611,
        filterLayout: 'oihw',
        inputLayout: 'nchw',
        label: 'Conv_84'
      }
    );
  
    
    const var_446 = builder.clamp(
      var_609,
      {
        minValue: 0,
        maxValue: 6
      }
    );
    
    const var_612 = builder.conv2d(
      var_446, var_613,
      {
        strides: [1, 1],
        padding: [1, 1, 1, 1],
        dilations: [1, 1],
        groups: 960,
        bias: var_614,
        filterLayout: 'oihw',
        inputLayout: 'nchw',
        label: 'Conv_86'
      }
    );
  
    
    const var_449 = builder.clamp(
      var_612,
      {
        minValue: 0,
        maxValue: 6
      }
    );
    
    const var_615 = builder.conv2d(
      var_449, var_616,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_617,
        filterLayout: 'oihw',
        inputLayout: 'nchw',
        label: 'Conv_88'
      }
    );
  
    
    const var_452 = builder.add(
      var_443,
      var_615
    );
    
    const var_618 = builder.conv2d(
      var_452, var_619,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_620,
        filterLayout: 'oihw',
        inputLayout: 'nchw',
        label: 'Conv_90'
      }
    );
  
    
    const var_455 = builder.clamp(
      var_618,
      {
        minValue: 0,
        maxValue: 6
      }
    );
    
    const var_621 = builder.conv2d(
      var_455, var_622,
      {
        strides: [1, 1],
        padding: [1, 1, 1, 1],
        dilations: [1, 1],
        groups: 960,
        bias: var_623,
        filterLayout: 'oihw',
        inputLayout: 'nchw',
        label: 'Conv_92'
      }
    );
  
    
    const var_458 = builder.clamp(
      var_621,
      {
        minValue: 0,
        maxValue: 6
      }
    );
    
    const var_624 = builder.conv2d(
      var_458, var_625,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_626,
        filterLayout: 'oihw',
        inputLayout: 'nchw',
        label: 'Conv_94'
      }
    );
  
    
    const var_627 = builder.conv2d(
      var_624, var_628,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_629,
        filterLayout: 'oihw',
        inputLayout: 'nchw',
        label: 'Conv_95'
      }
    );
  
    
    const var_463 = builder.clamp(
      var_627,
      {
        minValue: 0,
        maxValue: 6
      }
    );
    
    const var_464 = builder.averagePool2d(
      var_463
    );
    
    const var_472 = builder.reshape(
      var_464,
      (() => {
        const shape = Array.from(new BigInt64Array(weights_array_buffer, 8827824, 16 / BigInt64Array.BYTES_PER_ELEMENT), Number);
        // Calculate the concrete size for value -1.
        if (shape.includes(-1)) {
          const count = shape.filter(v => v === -1).length;
          if (count !== 1) {
            throw new Error('Only one -1 is allowed in reshape shape');
          }
          const totalInput = var_464.shape.reduce((a, b) => a * b, 1);
          const known = shape.reduce((a, b) => b === -1 ? a : a * b, 1);
          const idx = shape.indexOf(-1);
          shape[idx] = totalInput / known;
        }
        return shape;
      })()
    );
    
    const output = builder.gemm(
      var_472,
      var_classifier_1_weight,
      {
        alpha: 1.0,
        beta: 1.0,
        aTranspose: false,
        bTranspose: true,
        C: var_classifier_1_bias
      }
    );


    // Build graph with all outputs
    
    this.graph_ = await builder.build({ 'output': output });

    // Create output tensors
    
    this.outputTensors_['output'] = await this.context_.createTensor(
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