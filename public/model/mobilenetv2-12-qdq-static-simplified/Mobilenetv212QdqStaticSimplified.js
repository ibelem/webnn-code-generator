// WebNN Code Generator
// Todo: NCHW, NHWC layouts for BatchNormalization, InstanceNormalization, Conv, ConvInteger, 
// QLinearConv, ConvTranspose, AveragePool, LpPool, MaxPool, MaxUnpool, GlobalAveragePool, 
// GlobalLpPool, GlobalMaxPool, LRN, GridSample, DepthToSpace, SpaceToDepth

export class Mobilenetv212QdqStaticSimplified {

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
    
    const input = builder.input('input', { dataType: 'float32', shape: [1,3,224,224] });
    this.inputTensors_['input'] = await this.context_.createTensor(
      { dataType: 'float32', shape: [1,3,224,224], writable: true }
    );

    // Create graph constant operands
    
    const input_scale = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0.01865844801068306])
    );

    const input_zero_point = builder.constant(
      { dataType: 'uint8', shape: [] },
      new Uint8Array([114])
    );

    const var_475_quantized = builder.constant(
      { dataType: 'int8', shape: [32,3,3,3] },
      new Int8Array(weights_array_buffer.slice(10, 874))
    );
    
    const var_475_scale = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0.0028556426987051964])
    );

    const var_475_zero_point = builder.constant(
      { dataType: 'int8', shape: [] },
      new Int8Array([0])
    );

    const var_476_quantized = builder.constant(
      { dataType: 'int32', shape: [32] },
      new Int32Array(weights_array_buffer.slice(879, 1007))
    );
    
    const var_476_scale = builder.constant(
      { dataType: 'float32', shape: [1] },
      new Float32Array(weights_array_buffer.slice(1007, 1011))
    );
    
    const var_623_zero_point = builder.constant(
      { dataType: 'int32', shape: [1] },
      new Int32Array(weights_array_buffer.slice(2259276, 2259280))
    );
    
    const var_474_scale = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0.017607755959033966])
    );

    const var_474_zero_point = builder.constant(
      { dataType: 'uint8', shape: [] },
      new Uint8Array([0])
    );

    const var_478_quantized = builder.constant(
      { dataType: 'int8', shape: [32,1,3,3] },
      new Int8Array(weights_array_buffer.slice(1025, 1313))
    );
    
    const var_478_scale = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0.11868368089199066])
    );

    const var_479_quantized = builder.constant(
      { dataType: 'int32', shape: [32] },
      new Int32Array(weights_array_buffer.slice(1318, 1446))
    );
    
    const var_479_scale = builder.constant(
      { dataType: 'float32', shape: [1] },
      new Float32Array(weights_array_buffer.slice(1446, 1450))
    );
    
    const var_576_scale = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0.0235294122248888])
    );

    const var_481_quantized = builder.constant(
      { dataType: 'int8', shape: [16,32,1,1] },
      new Int8Array(weights_array_buffer.slice(1464, 1976))
    );
    
    const var_481_scale = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0.008153032511472702])
    );

    const var_482_quantized = builder.constant(
      { dataType: 'int32', shape: [16] },
      new Int32Array(weights_array_buffer.slice(1981, 2045))
    );
    
    const var_482_scale = builder.constant(
      { dataType: 'float32', shape: [1] },
      new Float32Array(weights_array_buffer.slice(2045, 2049))
    );
    
    const var_480_scale = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0.06564796715974808])
    );

    const var_480_zero_point = builder.constant(
      { dataType: 'uint8', shape: [] },
      new Uint8Array([119])
    );

    const var_484_quantized = builder.constant(
      { dataType: 'int8', shape: [96,16,1,1] },
      new Int8Array(weights_array_buffer.slice(2063, 3599))
    );
    
    const var_484_scale = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0.004745323210954666])
    );

    const var_485_quantized = builder.constant(
      { dataType: 'int32', shape: [96] },
      new Int32Array(weights_array_buffer.slice(3604, 3988))
    );
    
    const var_485_scale = builder.constant(
      { dataType: 'float32', shape: [1] },
      new Float32Array(weights_array_buffer.slice(3988, 3992))
    );
    
    const var_487_quantized = builder.constant(
      { dataType: 'int8', shape: [96,1,3,3] },
      new Int8Array(weights_array_buffer.slice(4006, 4870))
    );
    
    const var_487_scale = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0.050063420087099075])
    );

    const var_488_quantized = builder.constant(
      { dataType: 'int32', shape: [96] },
      new Int32Array(weights_array_buffer.slice(4875, 5259))
    );
    
    const var_488_scale = builder.constant(
      { dataType: 'float32', shape: [1] },
      new Float32Array(weights_array_buffer.slice(5259, 5263))
    );
    
    const var_490_quantized = builder.constant(
      { dataType: 'int8', shape: [24,96,1,1] },
      new Int8Array(weights_array_buffer.slice(5277, 7581))
    );
    
    const var_490_scale = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0.005946148186922073])
    );

    const var_491_quantized = builder.constant(
      { dataType: 'int32', shape: [24] },
      new Int32Array(weights_array_buffer.slice(7586, 7682))
    );
    
    const var_491_scale = builder.constant(
      { dataType: 'float32', shape: [1] },
      new Float32Array(weights_array_buffer.slice(7682, 7686))
    );
    
    const var_489_scale = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0.039710190147161484])
    );

    const var_489_zero_point = builder.constant(
      { dataType: 'uint8', shape: [] },
      new Uint8Array([107])
    );

    const var_493_quantized = builder.constant(
      { dataType: 'int8', shape: [144,24,1,1] },
      new Int8Array(weights_array_buffer.slice(7700, 11156))
    );
    
    const var_493_scale = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0.0026336642913520336])
    );

    const var_494_quantized = builder.constant(
      { dataType: 'int32', shape: [144] },
      new Int32Array(weights_array_buffer.slice(11161, 11737))
    );
    
    const var_494_scale = builder.constant(
      { dataType: 'float32', shape: [1] },
      new Float32Array(weights_array_buffer.slice(11737, 11741))
    );
    
    const var_492_scale = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0.009025856852531433])
    );

    const var_496_quantized = builder.constant(
      { dataType: 'int8', shape: [144,1,3,3] },
      new Int8Array(weights_array_buffer.slice(11755, 13051))
    );
    
    const var_496_scale = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0.03783246502280235])
    );

    const var_497_quantized = builder.constant(
      { dataType: 'int32', shape: [144] },
      new Int32Array(weights_array_buffer.slice(13056, 13632))
    );
    
    const var_497_scale = builder.constant(
      { dataType: 'float32', shape: [1] },
      new Float32Array(weights_array_buffer.slice(13632, 13636))
    );
    
    const var_495_scale = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0.019913742318749428])
    );

    const var_499_quantized = builder.constant(
      { dataType: 'int8', shape: [24,144,1,1] },
      new Int8Array(weights_array_buffer.slice(13650, 17106))
    );
    
    const var_499_scale = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0.00954227615147829])
    );

    const var_500_quantized = builder.constant(
      { dataType: 'int32', shape: [24] },
      new Int32Array(weights_array_buffer.slice(17111, 17207))
    );
    
    const var_500_scale = builder.constant(
      { dataType: 'float32', shape: [1] },
      new Float32Array(weights_array_buffer.slice(17207, 17211))
    );
    
    const var_498_scale = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0.04609981179237366])
    );

    const var_498_zero_point = builder.constant(
      { dataType: 'uint8', shape: [] },
      new Uint8Array([127])
    );

    const var_339_scale = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0.05797167494893074])
    );

    const var_339_zero_point = builder.constant(
      { dataType: 'uint8', shape: [] },
      new Uint8Array([125])
    );

    const var_502_quantized = builder.constant(
      { dataType: 'int8', shape: [144,24,1,1] },
      new Int8Array(weights_array_buffer.slice(17235, 20691))
    );
    
    const var_502_scale = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0.0024801704566925764])
    );

    const var_503_quantized = builder.constant(
      { dataType: 'int32', shape: [144] },
      new Int32Array(weights_array_buffer.slice(20696, 21272))
    );
    
    const var_503_scale = builder.constant(
      { dataType: 'float32', shape: [1] },
      new Float32Array(weights_array_buffer.slice(21272, 21276))
    );
    
    const var_501_scale = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0.013438905589282513])
    );

    const var_505_quantized = builder.constant(
      { dataType: 'int8', shape: [144,1,3,3] },
      new Int8Array(weights_array_buffer.slice(21290, 22586))
    );
    
    const var_505_scale = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0.046499304473400116])
    );

    const var_506_quantized = builder.constant(
      { dataType: 'int32', shape: [144] },
      new Int32Array(weights_array_buffer.slice(22591, 23167))
    );
    
    const var_506_scale = builder.constant(
      { dataType: 'float32', shape: [1] },
      new Float32Array(weights_array_buffer.slice(23167, 23171))
    );
    
    const var_504_scale = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0.014856177382171154])
    );

    const var_508_quantized = builder.constant(
      { dataType: 'int8', shape: [32,144,1,1] },
      new Int8Array(weights_array_buffer.slice(23185, 27793))
    );
    
    const var_508_scale = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0.006835097912698984])
    );

    const var_509_quantized = builder.constant(
      { dataType: 'int32', shape: [32] },
      new Int32Array(weights_array_buffer.slice(27798, 27926))
    );
    
    const var_509_scale = builder.constant(
      { dataType: 'float32', shape: [1] },
      new Float32Array(weights_array_buffer.slice(27926, 27930))
    );
    
    const var_507_scale = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0.03556208312511444])
    );

    const var_507_zero_point = builder.constant(
      { dataType: 'uint8', shape: [] },
      new Uint8Array([132])
    );

    const var_511_quantized = builder.constant(
      { dataType: 'int8', shape: [192,32,1,1] },
      new Int8Array(weights_array_buffer.slice(27944, 34088))
    );
    
    const var_511_scale = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0.0015136110596358776])
    );

    const var_512_quantized = builder.constant(
      { dataType: 'int32', shape: [192] },
      new Int32Array(weights_array_buffer.slice(34093, 34861))
    );
    
    const var_512_scale = builder.constant(
      { dataType: 'float32', shape: [1] },
      new Float32Array(weights_array_buffer.slice(34861, 34865))
    );
    
    const var_510_scale = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0.00862774346023798])
    );

    const var_514_quantized = builder.constant(
      { dataType: 'int8', shape: [192,1,3,3] },
      new Int8Array(weights_array_buffer.slice(34879, 36607))
    );
    
    const var_514_scale = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0.0517183281481266])
    );

    const var_515_quantized = builder.constant(
      { dataType: 'int32', shape: [192] },
      new Int32Array(weights_array_buffer.slice(36612, 37380))
    );
    
    const var_515_scale = builder.constant(
      { dataType: 'float32', shape: [1] },
      new Float32Array(weights_array_buffer.slice(37380, 37384))
    );
    
    const var_513_scale = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0.012746858410537243])
    );

    const var_517_quantized = builder.constant(
      { dataType: 'int8', shape: [32,192,1,1] },
      new Int8Array(weights_array_buffer.slice(37398, 43542))
    );
    
    const var_517_scale = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0.00651885848492384])
    );

    const var_518_quantized = builder.constant(
      { dataType: 'int32', shape: [32] },
      new Int32Array(weights_array_buffer.slice(43547, 43675))
    );
    
    const var_518_scale = builder.constant(
      { dataType: 'float32', shape: [1] },
      new Float32Array(weights_array_buffer.slice(43675, 43679))
    );
    
    const var_516_scale = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0.028934277594089508])
    );

    const var_516_zero_point = builder.constant(
      { dataType: 'uint8', shape: [] },
      new Uint8Array([128])
    );

    const var_356_scale = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0.04353522136807442])
    );

    const var_356_zero_point = builder.constant(
      { dataType: 'uint8', shape: [] },
      new Uint8Array([108])
    );

    const var_520_quantized = builder.constant(
      { dataType: 'int8', shape: [192,32,1,1] },
      new Int8Array(weights_array_buffer.slice(43703, 49847))
    );
    
    const var_520_scale = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0.0012169178808107972])
    );

    const var_521_quantized = builder.constant(
      { dataType: 'int32', shape: [192] },
      new Int32Array(weights_array_buffer.slice(49852, 50620))
    );
    
    const var_521_scale = builder.constant(
      { dataType: 'float32', shape: [1] },
      new Float32Array(weights_array_buffer.slice(50620, 50624))
    );
    
    const var_519_scale = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0.0076584890484809875])
    );

    const var_523_quantized = builder.constant(
      { dataType: 'int8', shape: [192,1,3,3] },
      new Int8Array(weights_array_buffer.slice(50638, 52366))
    );
    
    const var_523_scale = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0.03970986604690552])
    );

    const var_524_quantized = builder.constant(
      { dataType: 'int32', shape: [192] },
      new Int32Array(weights_array_buffer.slice(52371, 53139))
    );
    
    const var_524_scale = builder.constant(
      { dataType: 'float32', shape: [1] },
      new Float32Array(weights_array_buffer.slice(53139, 53143))
    );
    
    const var_522_scale = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0.01057371310889721])
    );

    const var_526_quantized = builder.constant(
      { dataType: 'int8', shape: [32,192,1,1] },
      new Int8Array(weights_array_buffer.slice(53157, 59301))
    );
    
    const var_526_scale = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0.007441135589033365])
    );

    const var_527_quantized = builder.constant(
      { dataType: 'int32', shape: [32] },
      new Int32Array(weights_array_buffer.slice(59306, 59434))
    );
    
    const var_527_scale = builder.constant(
      { dataType: 'float32', shape: [1] },
      new Float32Array(weights_array_buffer.slice(59434, 59438))
    );
    
    const var_525_scale = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0.02606847509741783])
    );

    const var_525_zero_point = builder.constant(
      { dataType: 'uint8', shape: [] },
      new Uint8Array([126])
    );

    const var_365_scale = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0.04982887580990791])
    );

    const var_365_zero_point = builder.constant(
      { dataType: 'uint8', shape: [] },
      new Uint8Array([120])
    );

    const var_529_quantized = builder.constant(
      { dataType: 'int8', shape: [192,32,1,1] },
      new Int8Array(weights_array_buffer.slice(59462, 65606))
    );
    
    const var_529_scale = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0.0017131756758317351])
    );

    const var_530_quantized = builder.constant(
      { dataType: 'int32', shape: [192] },
      new Int32Array(weights_array_buffer.slice(65611, 66379))
    );
    
    const var_530_scale = builder.constant(
      { dataType: 'float32', shape: [1] },
      new Float32Array(weights_array_buffer.slice(66379, 66383))
    );
    
    const var_528_scale = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0.011388028971850872])
    );

    const var_532_quantized = builder.constant(
      { dataType: 'int8', shape: [192,1,3,3] },
      new Int8Array(weights_array_buffer.slice(66397, 68125))
    );
    
    const var_532_scale = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0.01929706148803234])
    );

    const var_533_quantized = builder.constant(
      { dataType: 'int32', shape: [192] },
      new Int32Array(weights_array_buffer.slice(68130, 68898))
    );
    
    const var_533_scale = builder.constant(
      { dataType: 'float32', shape: [1] },
      new Float32Array(weights_array_buffer.slice(68898, 68902))
    );
    
    const var_531_scale = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0.014233097434043884])
    );

    const var_535_quantized = builder.constant(
      { dataType: 'int8', shape: [64,192,1,1] },
      new Int8Array(weights_array_buffer.slice(68916, 81204))
    );
    
    const var_535_scale = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0.005443144589662552])
    );

    const var_536_quantized = builder.constant(
      { dataType: 'int32', shape: [64] },
      new Int32Array(weights_array_buffer.slice(81209, 81465))
    );
    
    const var_536_scale = builder.constant(
      { dataType: 'float32', shape: [1] },
      new Float32Array(weights_array_buffer.slice(81465, 81469))
    );
    
    const var_534_scale = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0.028401868417859077])
    );

    const var_534_zero_point = builder.constant(
      { dataType: 'uint8', shape: [] },
      new Uint8Array([143])
    );

    const var_538_quantized = builder.constant(
      { dataType: 'int8', shape: [384,64,1,1] },
      new Int8Array(weights_array_buffer.slice(81483, 106059))
    );
    
    const var_538_scale = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0.0011173795210197568])
    );

    const var_539_quantized = builder.constant(
      { dataType: 'int32', shape: [384] },
      new Int32Array(weights_array_buffer.slice(106064, 107600))
    );
    
    const var_539_scale = builder.constant(
      { dataType: 'float32', shape: [1] },
      new Float32Array(weights_array_buffer.slice(107600, 107604))
    );
    
    const var_537_scale = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0.004354120697826147])
    );

    const var_541_quantized = builder.constant(
      { dataType: 'int8', shape: [384,1,3,3] },
      new Int8Array(weights_array_buffer.slice(107618, 111074))
    );
    
    const var_541_scale = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0.043541185557842255])
    );

    const var_542_quantized = builder.constant(
      { dataType: 'int32', shape: [384] },
      new Int32Array(weights_array_buffer.slice(111079, 112615))
    );
    
    const var_542_scale = builder.constant(
      { dataType: 'float32', shape: [1] },
      new Float32Array(weights_array_buffer.slice(112615, 112619))
    );
    
    const var_540_scale = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0.006769552361220121])
    );

    const var_544_quantized = builder.constant(
      { dataType: 'int8', shape: [64,384,1,1] },
      new Int8Array(weights_array_buffer.slice(112633, 137209))
    );
    
    const var_544_scale = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0.005489403381943703])
    );

    const var_545_quantized = builder.constant(
      { dataType: 'int32', shape: [64] },
      new Int32Array(weights_array_buffer.slice(137214, 137470))
    );
    
    const var_545_scale = builder.constant(
      { dataType: 'float32', shape: [1] },
      new Float32Array(weights_array_buffer.slice(137470, 137474))
    );
    
    const var_543_scale = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0.019568178802728653])
    );

    const var_543_zero_point = builder.constant(
      { dataType: 'uint8', shape: [] },
      new Uint8Array([130])
    );

    const var_382_scale = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0.02987072803080082])
    );

    const var_382_zero_point = builder.constant(
      { dataType: 'uint8', shape: [] },
      new Uint8Array([136])
    );

    const var_547_quantized = builder.constant(
      { dataType: 'int8', shape: [384,64,1,1] },
      new Int8Array(weights_array_buffer.slice(137498, 162074))
    );
    
    const var_547_scale = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0.0008492199121974409])
    );

    const var_548_quantized = builder.constant(
      { dataType: 'int32', shape: [384] },
      new Int32Array(weights_array_buffer.slice(162079, 163615))
    );
    
    const var_548_scale = builder.constant(
      { dataType: 'float32', shape: [1] },
      new Float32Array(weights_array_buffer.slice(163615, 163619))
    );
    
    const var_546_scale = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0.0040292879566550255])
    );

    const var_550_quantized = builder.constant(
      { dataType: 'int8', shape: [384,1,3,3] },
      new Int8Array(weights_array_buffer.slice(163633, 167089))
    );
    
    const var_550_scale = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0.08897167444229126])
    );

    const var_551_quantized = builder.constant(
      { dataType: 'int32', shape: [384] },
      new Int32Array(weights_array_buffer.slice(167094, 168630))
    );
    
    const var_551_scale = builder.constant(
      { dataType: 'float32', shape: [1] },
      new Float32Array(weights_array_buffer.slice(168630, 168634))
    );
    
    const var_549_scale = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0.006989649031311274])
    );

    const var_553_quantized = builder.constant(
      { dataType: 'int8', shape: [64,384,1,1] },
      new Int8Array(weights_array_buffer.slice(168648, 193224))
    );
    
    const var_553_scale = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0.005529865622520447])
    );

    const var_554_quantized = builder.constant(
      { dataType: 'int32', shape: [64] },
      new Int32Array(weights_array_buffer.slice(193229, 193485))
    );
    
    const var_554_scale = builder.constant(
      { dataType: 'float32', shape: [1] },
      new Float32Array(weights_array_buffer.slice(193485, 193489))
    );
    
    const var_552_scale = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0.01685042306780815])
    );

    const var_391_scale = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0.03107801266014576])
    );

    const var_556_quantized = builder.constant(
      { dataType: 'int8', shape: [384,64,1,1] },
      new Int8Array(weights_array_buffer.slice(193513, 218089))
    );
    
    const var_556_scale = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0.0008293222053907812])
    );

    const var_557_quantized = builder.constant(
      { dataType: 'int32', shape: [384] },
      new Int32Array(weights_array_buffer.slice(218094, 219630))
    );
    
    const var_557_scale = builder.constant(
      { dataType: 'float32', shape: [1] },
      new Float32Array(weights_array_buffer.slice(219630, 219634))
    );
    
    const var_555_scale = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0.0050100889056921005])
    );

    const var_559_quantized = builder.constant(
      { dataType: 'int8', shape: [384,1,3,3] },
      new Int8Array(weights_array_buffer.slice(219648, 223104))
    );
    
    const var_559_scale = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0.039976026862859726])
    );

    const var_560_quantized = builder.constant(
      { dataType: 'int32', shape: [384] },
      new Int32Array(weights_array_buffer.slice(223109, 224645))
    );
    
    const var_560_scale = builder.constant(
      { dataType: 'float32', shape: [1] },
      new Float32Array(weights_array_buffer.slice(224645, 224649))
    );
    
    const var_558_scale = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0.021815652027726173])
    );

    const var_562_quantized = builder.constant(
      { dataType: 'int8', shape: [64,384,1,1] },
      new Int8Array(weights_array_buffer.slice(224663, 249239))
    );
    
    const var_562_scale = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0.006337489001452923])
    );

    const var_563_quantized = builder.constant(
      { dataType: 'int32', shape: [64] },
      new Int32Array(weights_array_buffer.slice(249244, 249500))
    );
    
    const var_563_scale = builder.constant(
      { dataType: 'float32', shape: [1] },
      new Float32Array(weights_array_buffer.slice(249500, 249504))
    );
    
    const var_561_scale = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0.03308645263314247])
    );

    const var_561_zero_point = builder.constant(
      { dataType: 'uint8', shape: [] },
      new Uint8Array([160])
    );

    const var_400_scale = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0.037365980446338654])
    );

    const var_400_zero_point = builder.constant(
      { dataType: 'uint8', shape: [] },
      new Uint8Array([139])
    );

    const var_565_quantized = builder.constant(
      { dataType: 'int8', shape: [384,64,1,1] },
      new Int8Array(weights_array_buffer.slice(249528, 274104))
    );
    
    const var_565_scale = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0.0011116444366052747])
    );

    const var_566_quantized = builder.constant(
      { dataType: 'int32', shape: [384] },
      new Int32Array(weights_array_buffer.slice(274109, 275645))
    );
    
    const var_566_scale = builder.constant(
      { dataType: 'float32', shape: [1] },
      new Float32Array(weights_array_buffer.slice(275645, 275649))
    );
    
    const var_564_scale = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0.00675777904689312])
    );

    const var_568_quantized = builder.constant(
      { dataType: 'int8', shape: [384,1,3,3] },
      new Int8Array(weights_array_buffer.slice(275663, 279119))
    );
    
    const var_568_scale = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0.051308464258909225])
    );

    const var_569_quantized = builder.constant(
      { dataType: 'int32', shape: [384] },
      new Int32Array(weights_array_buffer.slice(279124, 280660))
    );
    
    const var_569_scale = builder.constant(
      { dataType: 'float32', shape: [1] },
      new Float32Array(weights_array_buffer.slice(280660, 280664))
    );
    
    const var_567_scale = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0.012513305060565472])
    );

    const var_571_quantized = builder.constant(
      { dataType: 'int8', shape: [96,384,1,1] },
      new Int8Array(weights_array_buffer.slice(280678, 317542))
    );
    
    const var_571_scale = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0.00487904017791152])
    );

    const var_572_quantized = builder.constant(
      { dataType: 'int32', shape: [96] },
      new Int32Array(weights_array_buffer.slice(317547, 317931))
    );
    
    const var_572_scale = builder.constant(
      { dataType: 'float32', shape: [1] },
      new Float32Array(weights_array_buffer.slice(317931, 317935))
    );
    
    const var_570_scale = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0.02412358671426773])
    );

    const var_624_zero_point = builder.constant(
      { dataType: 'uint8', shape: [] },
      new Uint8Array([123])
    );

    const var_574_quantized = builder.constant(
      { dataType: 'int8', shape: [576,96,1,1] },
      new Int8Array(weights_array_buffer.slice(317949, 373245))
    );
    
    const var_574_scale = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0.0013143877731636167])
    );

    const var_575_quantized = builder.constant(
      { dataType: 'int32', shape: [576] },
      new Int32Array(weights_array_buffer.slice(373250, 375554))
    );
    
    const var_575_scale = builder.constant(
      { dataType: 'float32', shape: [1] },
      new Float32Array(weights_array_buffer.slice(375554, 375558))
    );
    
    const var_573_scale = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0.006710308603942394])
    );

    const var_577_quantized = builder.constant(
      { dataType: 'int8', shape: [576,1,3,3] },
      new Int8Array(weights_array_buffer.slice(375572, 380756))
    );
    
    const var_577_scale = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0.07777220755815506])
    );

    const var_578_quantized = builder.constant(
      { dataType: 'int32', shape: [576] },
      new Int32Array(weights_array_buffer.slice(380761, 383065))
    );
    
    const var_578_scale = builder.constant(
      { dataType: 'float32', shape: [1] },
      new Float32Array(weights_array_buffer.slice(383065, 383069))
    );
    
    const var_580_quantized = builder.constant(
      { dataType: 'int8', shape: [96,576,1,1] },
      new Int8Array(weights_array_buffer.slice(383083, 438379))
    );
    
    const var_580_scale = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0.003954809159040451])
    );

    const var_581_quantized = builder.constant(
      { dataType: 'int32', shape: [96] },
      new Int32Array(weights_array_buffer.slice(438384, 438768))
    );
    
    const var_581_scale = builder.constant(
      { dataType: 'float32', shape: [1] },
      new Float32Array(weights_array_buffer.slice(438768, 438772))
    );
    
    const var_579_scale = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0.027985379099845886])
    );

    const var_579_zero_point = builder.constant(
      { dataType: 'uint8', shape: [] },
      new Uint8Array([115])
    );

    const var_417_scale = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0.03182479739189148])
    );

    const var_417_zero_point = builder.constant(
      { dataType: 'uint8', shape: [] },
      new Uint8Array([109])
    );

    const var_583_quantized = builder.constant(
      { dataType: 'int8', shape: [576,96,1,1] },
      new Int8Array(weights_array_buffer.slice(438796, 494092))
    );
    
    const var_583_scale = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0.001712543424218893])
    );

    const var_584_quantized = builder.constant(
      { dataType: 'int32', shape: [576] },
      new Int32Array(weights_array_buffer.slice(494097, 496401))
    );
    
    const var_584_scale = builder.constant(
      { dataType: 'float32', shape: [1] },
      new Float32Array(weights_array_buffer.slice(496401, 496405))
    );
    
    const var_582_scale = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0.01857798360288143])
    );

    const var_586_quantized = builder.constant(
      { dataType: 'int8', shape: [576,1,3,3] },
      new Int8Array(weights_array_buffer.slice(496419, 501603))
    );
    
    const var_586_scale = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0.06750054657459259])
    );

    const var_587_quantized = builder.constant(
      { dataType: 'int32', shape: [576] },
      new Int32Array(weights_array_buffer.slice(501608, 503912))
    );
    
    const var_587_scale = builder.constant(
      { dataType: 'float32', shape: [1] },
      new Float32Array(weights_array_buffer.slice(503912, 503916))
    );
    
    const var_589_quantized = builder.constant(
      { dataType: 'int8', shape: [96,576,1,1] },
      new Int8Array(weights_array_buffer.slice(503930, 559226))
    );
    
    const var_589_scale = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0.009840592741966248])
    );

    const var_590_quantized = builder.constant(
      { dataType: 'int32', shape: [96] },
      new Int32Array(weights_array_buffer.slice(559231, 559615))
    );
    
    const var_590_scale = builder.constant(
      { dataType: 'float32', shape: [1] },
      new Float32Array(weights_array_buffer.slice(559615, 559619))
    );
    
    const var_588_scale = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0.06331735104322433])
    );

    const var_588_zero_point = builder.constant(
      { dataType: 'uint8', shape: [] },
      new Uint8Array([142])
    );

    const var_426_scale = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0.07048992812633514])
    );

    const var_592_quantized = builder.constant(
      { dataType: 'int8', shape: [576,96,1,1] },
      new Int8Array(weights_array_buffer.slice(559643, 614939))
    );
    
    const var_592_scale = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0.0014183413004502654])
    );

    const var_593_quantized = builder.constant(
      { dataType: 'int32', shape: [576] },
      new Int32Array(weights_array_buffer.slice(614944, 617248))
    );
    
    const var_593_scale = builder.constant(
      { dataType: 'float32', shape: [1] },
      new Float32Array(weights_array_buffer.slice(617248, 617252))
    );
    
    const var_591_scale = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0.014397864229977131])
    );

    const var_595_quantized = builder.constant(
      { dataType: 'int8', shape: [576,1,3,3] },
      new Int8Array(weights_array_buffer.slice(617266, 622450))
    );
    
    const var_595_scale = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0.01818242482841015])
    );

    const var_596_quantized = builder.constant(
      { dataType: 'int32', shape: [576] },
      new Int32Array(weights_array_buffer.slice(622455, 624759))
    );
    
    const var_596_scale = builder.constant(
      { dataType: 'float32', shape: [1] },
      new Float32Array(weights_array_buffer.slice(624759, 624763))
    );
    
    const var_598_quantized = builder.constant(
      { dataType: 'int8', shape: [160,576,1,1] },
      new Int8Array(weights_array_buffer.slice(624777, 716937))
    );
    
    const var_598_scale = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0.0025079844053834677])
    );

    const var_599_quantized = builder.constant(
      { dataType: 'int32', shape: [160] },
      new Int32Array(weights_array_buffer.slice(716942, 717582))
    );
    
    const var_599_scale = builder.constant(
      { dataType: 'float32', shape: [1] },
      new Float32Array(weights_array_buffer.slice(717582, 717586))
    );
    
    const var_597_scale = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0.06929498165845871])
    );

    const var_597_zero_point = builder.constant(
      { dataType: 'uint8', shape: [] },
      new Uint8Array([149])
    );

    const var_601_quantized = builder.constant(
      { dataType: 'int8', shape: [960,160,1,1] },
      new Int8Array(weights_array_buffer.slice(717600, 871200))
    );
    
    const var_601_scale = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0.0029125262517482042])
    );

    const var_602_quantized = builder.constant(
      { dataType: 'int32', shape: [960] },
      new Int32Array(weights_array_buffer.slice(871205, 875045))
    );
    
    const var_602_scale = builder.constant(
      { dataType: 'float32', shape: [1] },
      new Float32Array(weights_array_buffer.slice(875045, 875049))
    );
    
    const var_600_scale = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0.02097153291106224])
    );

    const var_604_quantized = builder.constant(
      { dataType: 'int8', shape: [960,1,3,3] },
      new Int8Array(weights_array_buffer.slice(875063, 883703))
    );
    
    const var_604_scale = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0.07850213348865509])
    );

    const var_605_quantized = builder.constant(
      { dataType: 'int32', shape: [960] },
      new Int32Array(weights_array_buffer.slice(883708, 887548))
    );
    
    const var_605_scale = builder.constant(
      { dataType: 'float32', shape: [1] },
      new Float32Array(weights_array_buffer.slice(887548, 887552))
    );
    
    const var_603_scale = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0.022908082231879234])
    );

    const var_607_quantized = builder.constant(
      { dataType: 'int8', shape: [160,960,1,1] },
      new Int8Array(weights_array_buffer.slice(887566, 1041166))
    );
    
    const var_607_scale = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0.0025744689628481865])
    );

    const var_608_quantized = builder.constant(
      { dataType: 'int32', shape: [160] },
      new Int32Array(weights_array_buffer.slice(1041171, 1041811))
    );
    
    const var_608_scale = builder.constant(
      { dataType: 'float32', shape: [1] },
      new Float32Array(weights_array_buffer.slice(1041811, 1041815))
    );
    
    const var_606_scale = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0.06415010988712311])
    );

    const var_606_zero_point = builder.constant(
      { dataType: 'uint8', shape: [] },
      new Uint8Array([105])
    );

    const var_443_scale = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0.12168142944574356])
    );

    const var_610_quantized = builder.constant(
      { dataType: 'int8', shape: [960,160,1,1] },
      new Int8Array(weights_array_buffer.slice(1041839, 1195439))
    );
    
    const var_610_scale = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0.0016836411086842418])
    );

    const var_611_quantized = builder.constant(
      { dataType: 'int32', shape: [960] },
      new Int32Array(weights_array_buffer.slice(1195444, 1199284))
    );
    
    const var_611_scale = builder.constant(
      { dataType: 'float32', shape: [1] },
      new Float32Array(weights_array_buffer.slice(1199284, 1199288))
    );
    
    const var_609_scale = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0.020093470811843872])
    );

    const var_613_quantized = builder.constant(
      { dataType: 'int8', shape: [960,1,3,3] },
      new Int8Array(weights_array_buffer.slice(1199302, 1207942))
    );
    
    const var_613_scale = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0.05770537257194519])
    );

    const var_614_quantized = builder.constant(
      { dataType: 'int32', shape: [960] },
      new Int32Array(weights_array_buffer.slice(1207947, 1211787))
    );
    
    const var_614_scale = builder.constant(
      { dataType: 'float32', shape: [1] },
      new Float32Array(weights_array_buffer.slice(1211787, 1211791))
    );
    
    const var_616_quantized = builder.constant(
      { dataType: 'int8', shape: [160,960,1,1] },
      new Int8Array(weights_array_buffer.slice(1211805, 1365405))
    );
    
    const var_616_scale = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0.004802136216312647])
    );

    const var_617_quantized = builder.constant(
      { dataType: 'int32', shape: [160] },
      new Int32Array(weights_array_buffer.slice(1365410, 1366050))
    );
    
    const var_617_scale = builder.constant(
      { dataType: 'float32', shape: [1] },
      new Float32Array(weights_array_buffer.slice(1366050, 1366054))
    );
    
    const var_615_scale = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0.14457149803638458])
    );

    const var_615_zero_point = builder.constant(
      { dataType: 'uint8', shape: [] },
      new Uint8Array([131])
    );

    const var_452_scale = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0.23722179234027863])
    );

    const var_452_zero_point = builder.constant(
      { dataType: 'uint8', shape: [] },
      new Uint8Array([110])
    );

    const var_619_quantized = builder.constant(
      { dataType: 'int8', shape: [960,160,1,1] },
      new Int8Array(weights_array_buffer.slice(1366078, 1519678))
    );
    
    const var_619_scale = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0.0010381987085565925])
    );

    const var_620_quantized = builder.constant(
      { dataType: 'int32', shape: [960] },
      new Int32Array(weights_array_buffer.slice(1519683, 1523523))
    );
    
    const var_620_scale = builder.constant(
      { dataType: 'float32', shape: [1] },
      new Float32Array(weights_array_buffer.slice(1523523, 1523527))
    );
    
    const var_618_scale = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0.015018466860055923])
    );

    const var_622_quantized = builder.constant(
      { dataType: 'int8', shape: [960,1,3,3] },
      new Int8Array(weights_array_buffer.slice(1523541, 1532181))
    );
    
    const var_622_scale = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0.07043053954839706])
    );

    const var_623_quantized = builder.constant(
      { dataType: 'int32', shape: [960] },
      new Int32Array(weights_array_buffer.slice(1532186, 1536026))
    );
    
    const var_623_scale = builder.constant(
      { dataType: 'float32', shape: [1] },
      new Float32Array(weights_array_buffer.slice(1536026, 1536030))
    );
    
    const var_621_scale = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0.008207708597183228])
    );

    const var_625_quantized = builder.constant(
      { dataType: 'int8', shape: [320,960,1,1] },
      new Int8Array(weights_array_buffer.slice(1536044, 1843244))
    );
    
    const var_625_scale = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0.005075970198959112])
    );

    const var_626_quantized = builder.constant(
      { dataType: 'int32', shape: [320] },
      new Int32Array(weights_array_buffer.slice(1843249, 1844529))
    );
    
    const var_626_scale = builder.constant(
      { dataType: 'float32', shape: [1] },
      new Float32Array(weights_array_buffer.slice(1844529, 1844533))
    );
    
    const var_624_scale = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0.01452525146305561])
    );

    const var_628_quantized = builder.constant(
      { dataType: 'int8', shape: [1280,320,1,1] },
      new Int8Array(weights_array_buffer.slice(1844547, 2254147))
    );
    
    const var_628_scale = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0.013814685866236687])
    );

    const var_629_quantized = builder.constant(
      { dataType: 'int32', shape: [1280] },
      new Int32Array(weights_array_buffer.slice(2254152, 2259272))
    );
    
    const var_629_scale = builder.constant(
      { dataType: 'float32', shape: [1] },
      new Float32Array(weights_array_buffer.slice(2259272, 2259276))
    );
    
    const var_464_scale = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0.019110053777694702])
    );

    const var_471 = builder.constant(
      { dataType: 'int64', shape: [2] },
      new BigInt64Array(weights_array_buffer.slice(2259300, 2259316))
    );
    
    const var_classifier_1_weight_quantized = builder.constant(
      { dataType: 'int8', shape: [1280,1000] },
      new Int8Array(weights_array_buffer.slice(2259326, 3539326))
    );
    
    const var_classifier_1_weight_scale = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0.0026049718726426363])
    );

    const output_matmul_scale = builder.constant(
      { dataType: 'float32', shape: [] },
      new Float32Array([0.16079935431480408])
    );

    const output_matmul_zero_point = builder.constant(
      { dataType: 'uint8', shape: [] },
      new Uint8Array([76])
    );

    const var_classifier_1_bias = builder.constant(
      { dataType: 'float32', shape: [1000] },
      new Float32Array(weights_array_buffer.slice(3539341, 3543341))
    );
    

    // Create graph operators
        
    const input_quantized = builder.quantizeLinear(
      input,
      builder.reshape(input_scale, [1, 1, 1, 1]),
      builder.reshape(input_zero_point, [1, 1, 1, 1])
    );
    
    const input_dequantized = builder.dequantizeLinear(
      input_quantized,
      builder.reshape(input_scale, [1, 1, 1, 1]),
      builder.reshape(input_zero_point, [1, 1, 1, 1])
    );
    
    const var_475_dequantized = builder.dequantizeLinear(
      var_475_quantized,
      builder.reshape(var_475_scale, [1, 1, 1, 1]),
      builder.reshape(var_475_zero_point, [1, 1, 1, 1])
    );
    
    const var_476_dequantized = builder.dequantizeLinear(
      var_476_quantized,
      var_476_scale,
      var_623_zero_point
    );
    
    const var_474_quantizeinput = builder.conv2d(
      input_dequantized, var_475_dequantized,
      {
        strides: [2, 2],
        padding: [1, 1, 1, 1],
        dilations: [1, 1],
        groups: 1,
        bias: var_476_dequantized,
        filterLayout: 'oihw',
        inputLayout: 'nchw'
      }
    );
    
    const var_474_quantized = builder.quantizeLinear(
      var_474_quantizeinput,
      builder.reshape(var_474_scale, [1, 1, 1, 1]),
      builder.reshape(var_474_zero_point, [1, 1, 1, 1])
    );
    
    const var_474_conv_2_dequantized = builder.dequantizeLinear(
      var_474_quantized,
      builder.reshape(var_474_scale, [1, 1, 1, 1]),
      builder.reshape(var_474_zero_point, [1, 1, 1, 1])
    );
    
    const var_478_dequantized = builder.dequantizeLinear(
      var_478_quantized,
      builder.reshape(var_478_scale, [1, 1, 1, 1]),
      builder.reshape(var_475_zero_point, [1, 1, 1, 1])
    );
    
    const var_479_dequantized = builder.dequantizeLinear(
      var_479_quantized,
      var_479_scale,
      var_623_zero_point
    );
    
    const var_477_quantizeinput = builder.conv2d(
      var_474_conv_2_dequantized, var_478_dequantized,
      {
        strides: [1, 1],
        padding: [1, 1, 1, 1],
        dilations: [1, 1],
        groups: 32,
        bias: var_479_dequantized,
        filterLayout: 'oihw',
        inputLayout: 'nchw'
      }
    );
    
    const var_477_quantized = builder.quantizeLinear(
      var_477_quantizeinput,
      builder.reshape(var_576_scale, [1, 1, 1, 1]),
      builder.reshape(var_474_zero_point, [1, 1, 1, 1])
    );
    
    const var_477_conv_4_dequantized = builder.dequantizeLinear(
      var_477_quantized,
      builder.reshape(var_576_scale, [1, 1, 1, 1]),
      builder.reshape(var_474_zero_point, [1, 1, 1, 1])
    );
    
    const var_481_dequantized = builder.dequantizeLinear(
      var_481_quantized,
      builder.reshape(var_481_scale, [1, 1, 1, 1]),
      builder.reshape(var_475_zero_point, [1, 1, 1, 1])
    );
    
    const var_482_dequantized = builder.dequantizeLinear(
      var_482_quantized,
      var_482_scale,
      var_623_zero_point
    );
    
    const var_480_quantizeinput = builder.conv2d(
      var_477_conv_4_dequantized, var_481_dequantized,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_482_dequantized,
        filterLayout: 'oihw',
        inputLayout: 'nchw'
      }
    );
    
    const var_480_quantized = builder.quantizeLinear(
      var_480_quantizeinput,
      builder.reshape(var_480_scale, [1, 1, 1, 1]),
      builder.reshape(var_480_zero_point, [1, 1, 1, 1])
    );
    
    const var_480_conv_5_dequantized = builder.dequantizeLinear(
      var_480_quantized,
      builder.reshape(var_480_scale, [1, 1, 1, 1]),
      builder.reshape(var_480_zero_point, [1, 1, 1, 1])
    );
    
    const var_484_dequantized = builder.dequantizeLinear(
      var_484_quantized,
      builder.reshape(var_484_scale, [1, 1, 1, 1]),
      builder.reshape(var_475_zero_point, [1, 1, 1, 1])
    );
    
    const var_485_dequantized = builder.dequantizeLinear(
      var_485_quantized,
      var_485_scale,
      var_623_zero_point
    );
    
    const var_483_quantizeinput = builder.conv2d(
      var_480_conv_5_dequantized, var_484_dequantized,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_485_dequantized,
        filterLayout: 'oihw',
        inputLayout: 'nchw'
      }
    );
    
    const var_483_quantized = builder.quantizeLinear(
      var_483_quantizeinput,
      builder.reshape(var_576_scale, [1, 1, 1, 1]),
      builder.reshape(var_474_zero_point, [1, 1, 1, 1])
    );
    
    const var_483_conv_7_dequantized = builder.dequantizeLinear(
      var_483_quantized,
      builder.reshape(var_576_scale, [1, 1, 1, 1]),
      builder.reshape(var_474_zero_point, [1, 1, 1, 1])
    );
    
    const var_487_dequantized = builder.dequantizeLinear(
      var_487_quantized,
      builder.reshape(var_487_scale, [1, 1, 1, 1]),
      builder.reshape(var_475_zero_point, [1, 1, 1, 1])
    );
    
    const var_488_dequantized = builder.dequantizeLinear(
      var_488_quantized,
      var_488_scale,
      var_623_zero_point
    );
    
    const var_486_quantizeinput = builder.conv2d(
      var_483_conv_7_dequantized, var_487_dequantized,
      {
        strides: [2, 2],
        padding: [1, 1, 1, 1],
        dilations: [1, 1],
        groups: 96,
        bias: var_488_dequantized,
        filterLayout: 'oihw',
        inputLayout: 'nchw'
      }
    );
    
    const var_486_quantized = builder.quantizeLinear(
      var_486_quantizeinput,
      builder.reshape(var_576_scale, [1, 1, 1, 1]),
      builder.reshape(var_474_zero_point, [1, 1, 1, 1])
    );
    
    const var_486_conv_9_dequantized = builder.dequantizeLinear(
      var_486_quantized,
      builder.reshape(var_576_scale, [1, 1, 1, 1]),
      builder.reshape(var_474_zero_point, [1, 1, 1, 1])
    );
    
    const var_490_dequantized = builder.dequantizeLinear(
      var_490_quantized,
      builder.reshape(var_490_scale, [1, 1, 1, 1]),
      builder.reshape(var_475_zero_point, [1, 1, 1, 1])
    );
    
    const var_491_dequantized = builder.dequantizeLinear(
      var_491_quantized,
      var_491_scale,
      var_623_zero_point
    );
    
    const var_489_quantizeinput = builder.conv2d(
      var_486_conv_9_dequantized, var_490_dequantized,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_491_dequantized,
        filterLayout: 'oihw',
        inputLayout: 'nchw'
      }
    );
    
    const var_489_quantized = builder.quantizeLinear(
      var_489_quantizeinput,
      builder.reshape(var_489_scale, [1, 1, 1, 1]),
      builder.reshape(var_489_zero_point, [1, 1, 1, 1])
    );
    
    const var_489_duplicated = builder.dequantizeLinear(
      var_489_quantized,
      builder.reshape(var_489_scale, [1, 1, 1, 1]),
      builder.reshape(var_489_zero_point, [1, 1, 1, 1])
    );
    
    const var_493_dequantized = builder.dequantizeLinear(
      var_493_quantized,
      builder.reshape(var_493_scale, [1, 1, 1, 1]),
      builder.reshape(var_475_zero_point, [1, 1, 1, 1])
    );
    
    const var_494_dequantized = builder.dequantizeLinear(
      var_494_quantized,
      var_494_scale,
      var_623_zero_point
    );
    
    const var_492_quantizeinput = builder.conv2d(
      var_489_duplicated, var_493_dequantized,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_494_dequantized,
        filterLayout: 'oihw',
        inputLayout: 'nchw'
      }
    );
    
    const var_492_quantized = builder.quantizeLinear(
      var_492_quantizeinput,
      builder.reshape(var_492_scale, [1, 1, 1, 1]),
      builder.reshape(var_474_zero_point, [1, 1, 1, 1])
    );
    
    const var_492_conv_12_dequantized = builder.dequantizeLinear(
      var_492_quantized,
      builder.reshape(var_492_scale, [1, 1, 1, 1]),
      builder.reshape(var_474_zero_point, [1, 1, 1, 1])
    );
    
    const var_496_dequantized = builder.dequantizeLinear(
      var_496_quantized,
      builder.reshape(var_496_scale, [1, 1, 1, 1]),
      builder.reshape(var_475_zero_point, [1, 1, 1, 1])
    );
    
    const var_497_dequantized = builder.dequantizeLinear(
      var_497_quantized,
      var_497_scale,
      var_623_zero_point
    );
    
    const var_495_quantizeinput = builder.conv2d(
      var_492_conv_12_dequantized, var_496_dequantized,
      {
        strides: [1, 1],
        padding: [1, 1, 1, 1],
        dilations: [1, 1],
        groups: 144,
        bias: var_497_dequantized,
        filterLayout: 'oihw',
        inputLayout: 'nchw'
      }
    );
    
    const var_495_quantized = builder.quantizeLinear(
      var_495_quantizeinput,
      builder.reshape(var_495_scale, [1, 1, 1, 1]),
      builder.reshape(var_474_zero_point, [1, 1, 1, 1])
    );
    
    const var_495_conv_14_dequantized = builder.dequantizeLinear(
      var_495_quantized,
      builder.reshape(var_495_scale, [1, 1, 1, 1]),
      builder.reshape(var_474_zero_point, [1, 1, 1, 1])
    );
    
    const var_499_dequantized = builder.dequantizeLinear(
      var_499_quantized,
      builder.reshape(var_499_scale, [1, 1, 1, 1]),
      builder.reshape(var_475_zero_point, [1, 1, 1, 1])
    );
    
    const var_500_dequantized = builder.dequantizeLinear(
      var_500_quantized,
      var_500_scale,
      var_623_zero_point
    );
    
    const var_498_quantizeinput = builder.conv2d(
      var_495_conv_14_dequantized, var_499_dequantized,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_500_dequantized,
        filterLayout: 'oihw',
        inputLayout: 'nchw'
      }
    );
    
    const var_498_quantized = builder.quantizeLinear(
      var_498_quantizeinput,
      builder.reshape(var_498_scale, [1, 1, 1, 1]),
      builder.reshape(var_498_zero_point, [1, 1, 1, 1])
    );
    
    const var_498 = builder.dequantizeLinear(
      var_498_quantized,
      builder.reshape(var_498_scale, [1, 1, 1, 1]),
      builder.reshape(var_498_zero_point, [1, 1, 1, 1])
    );
    
    const var_339 = builder.add(
      var_489_duplicated,
      var_498
    );
    
    const var_339_conv_16_quantizelinear = builder.quantizeLinear(
      var_339,
      builder.reshape(var_339_scale, [1, 1, 1, 1]),
      builder.reshape(var_339_zero_point, [1, 1, 1, 1])
    );
    
    const var_339_conv_16_dequantized = builder.dequantizeLinear(
      var_339_conv_16_quantizelinear,
      builder.reshape(var_339_scale, [1, 1, 1, 1]),
      builder.reshape(var_339_zero_point, [1, 1, 1, 1])
    );
    
    const var_502_dequantized = builder.dequantizeLinear(
      var_502_quantized,
      builder.reshape(var_502_scale, [1, 1, 1, 1]),
      builder.reshape(var_475_zero_point, [1, 1, 1, 1])
    );
    
    const var_503_dequantized = builder.dequantizeLinear(
      var_503_quantized,
      var_503_scale,
      var_623_zero_point
    );
    
    const var_501_quantizeinput = builder.conv2d(
      var_339_conv_16_dequantized, var_502_dequantized,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_503_dequantized,
        filterLayout: 'oihw',
        inputLayout: 'nchw'
      }
    );
    
    const var_501_quantized = builder.quantizeLinear(
      var_501_quantizeinput,
      builder.reshape(var_501_scale, [1, 1, 1, 1]),
      builder.reshape(var_474_zero_point, [1, 1, 1, 1])
    );
    
    const var_501_conv_18_dequantized = builder.dequantizeLinear(
      var_501_quantized,
      builder.reshape(var_501_scale, [1, 1, 1, 1]),
      builder.reshape(var_474_zero_point, [1, 1, 1, 1])
    );
    
    const var_505_dequantized = builder.dequantizeLinear(
      var_505_quantized,
      builder.reshape(var_505_scale, [1, 1, 1, 1]),
      builder.reshape(var_475_zero_point, [1, 1, 1, 1])
    );
    
    const var_506_dequantized = builder.dequantizeLinear(
      var_506_quantized,
      var_506_scale,
      var_623_zero_point
    );
    
    const var_504_quantizeinput = builder.conv2d(
      var_501_conv_18_dequantized, var_505_dequantized,
      {
        strides: [2, 2],
        padding: [1, 1, 1, 1],
        dilations: [1, 1],
        groups: 144,
        bias: var_506_dequantized,
        filterLayout: 'oihw',
        inputLayout: 'nchw'
      }
    );
    
    const var_504_quantized = builder.quantizeLinear(
      var_504_quantizeinput,
      builder.reshape(var_504_scale, [1, 1, 1, 1]),
      builder.reshape(var_474_zero_point, [1, 1, 1, 1])
    );
    
    const var_504_conv_20_dequantized = builder.dequantizeLinear(
      var_504_quantized,
      builder.reshape(var_504_scale, [1, 1, 1, 1]),
      builder.reshape(var_474_zero_point, [1, 1, 1, 1])
    );
    
    const var_508_dequantized = builder.dequantizeLinear(
      var_508_quantized,
      builder.reshape(var_508_scale, [1, 1, 1, 1]),
      builder.reshape(var_475_zero_point, [1, 1, 1, 1])
    );
    
    const var_509_dequantized = builder.dequantizeLinear(
      var_509_quantized,
      var_509_scale,
      var_623_zero_point
    );
    
    const var_507_quantizeinput = builder.conv2d(
      var_504_conv_20_dequantized, var_508_dequantized,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_509_dequantized,
        filterLayout: 'oihw',
        inputLayout: 'nchw'
      }
    );
    
    const var_507_quantized = builder.quantizeLinear(
      var_507_quantizeinput,
      builder.reshape(var_507_scale, [1, 1, 1, 1]),
      builder.reshape(var_507_zero_point, [1, 1, 1, 1])
    );
    
    const var_507_duplicated = builder.dequantizeLinear(
      var_507_quantized,
      builder.reshape(var_507_scale, [1, 1, 1, 1]),
      builder.reshape(var_507_zero_point, [1, 1, 1, 1])
    );
    
    const var_511_dequantized = builder.dequantizeLinear(
      var_511_quantized,
      builder.reshape(var_511_scale, [1, 1, 1, 1]),
      builder.reshape(var_475_zero_point, [1, 1, 1, 1])
    );
    
    const var_512_dequantized = builder.dequantizeLinear(
      var_512_quantized,
      var_512_scale,
      var_623_zero_point
    );
    
    const var_510_quantizeinput = builder.conv2d(
      var_507_duplicated, var_511_dequantized,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_512_dequantized,
        filterLayout: 'oihw',
        inputLayout: 'nchw'
      }
    );
    
    const var_510_quantized = builder.quantizeLinear(
      var_510_quantizeinput,
      builder.reshape(var_510_scale, [1, 1, 1, 1]),
      builder.reshape(var_474_zero_point, [1, 1, 1, 1])
    );
    
    const var_510_conv_23_dequantized = builder.dequantizeLinear(
      var_510_quantized,
      builder.reshape(var_510_scale, [1, 1, 1, 1]),
      builder.reshape(var_474_zero_point, [1, 1, 1, 1])
    );
    
    const var_514_dequantized = builder.dequantizeLinear(
      var_514_quantized,
      builder.reshape(var_514_scale, [1, 1, 1, 1]),
      builder.reshape(var_475_zero_point, [1, 1, 1, 1])
    );
    
    const var_515_dequantized = builder.dequantizeLinear(
      var_515_quantized,
      var_515_scale,
      var_623_zero_point
    );
    
    const var_513_quantizeinput = builder.conv2d(
      var_510_conv_23_dequantized, var_514_dequantized,
      {
        strides: [1, 1],
        padding: [1, 1, 1, 1],
        dilations: [1, 1],
        groups: 192,
        bias: var_515_dequantized,
        filterLayout: 'oihw',
        inputLayout: 'nchw'
      }
    );
    
    const var_513_quantized = builder.quantizeLinear(
      var_513_quantizeinput,
      builder.reshape(var_513_scale, [1, 1, 1, 1]),
      builder.reshape(var_474_zero_point, [1, 1, 1, 1])
    );
    
    const var_513_conv_25_dequantized = builder.dequantizeLinear(
      var_513_quantized,
      builder.reshape(var_513_scale, [1, 1, 1, 1]),
      builder.reshape(var_474_zero_point, [1, 1, 1, 1])
    );
    
    const var_517_dequantized = builder.dequantizeLinear(
      var_517_quantized,
      builder.reshape(var_517_scale, [1, 1, 1, 1]),
      builder.reshape(var_475_zero_point, [1, 1, 1, 1])
    );
    
    const var_518_dequantized = builder.dequantizeLinear(
      var_518_quantized,
      var_518_scale,
      var_623_zero_point
    );
    
    const var_516_quantizeinput = builder.conv2d(
      var_513_conv_25_dequantized, var_517_dequantized,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_518_dequantized,
        filterLayout: 'oihw',
        inputLayout: 'nchw'
      }
    );
    
    const var_516_quantized = builder.quantizeLinear(
      var_516_quantizeinput,
      builder.reshape(var_516_scale, [1, 1, 1, 1]),
      builder.reshape(var_516_zero_point, [1, 1, 1, 1])
    );
    
    const var_516 = builder.dequantizeLinear(
      var_516_quantized,
      builder.reshape(var_516_scale, [1, 1, 1, 1]),
      builder.reshape(var_516_zero_point, [1, 1, 1, 1])
    );
    
    const var_356 = builder.add(
      var_507_duplicated,
      var_516
    );
    
    const var_356_conv_27_quantizelinear = builder.quantizeLinear(
      var_356,
      builder.reshape(var_356_scale, [1, 1, 1, 1]),
      builder.reshape(var_356_zero_point, [1, 1, 1, 1])
    );
    
    const var_356_conv_27_dequantized = builder.dequantizeLinear(
      var_356_conv_27_quantizelinear,
      builder.reshape(var_356_scale, [1, 1, 1, 1]),
      builder.reshape(var_356_zero_point, [1, 1, 1, 1])
    );
    
    const var_520_dequantized = builder.dequantizeLinear(
      var_520_quantized,
      builder.reshape(var_520_scale, [1, 1, 1, 1]),
      builder.reshape(var_475_zero_point, [1, 1, 1, 1])
    );
    
    const var_521_dequantized = builder.dequantizeLinear(
      var_521_quantized,
      var_521_scale,
      var_623_zero_point
    );
    
    const var_519_quantizeinput = builder.conv2d(
      var_356_conv_27_dequantized, var_520_dequantized,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_521_dequantized,
        filterLayout: 'oihw',
        inputLayout: 'nchw'
      }
    );
    
    const var_519_quantized = builder.quantizeLinear(
      var_519_quantizeinput,
      builder.reshape(var_519_scale, [1, 1, 1, 1]),
      builder.reshape(var_474_zero_point, [1, 1, 1, 1])
    );
    
    const var_519_conv_29_dequantized = builder.dequantizeLinear(
      var_519_quantized,
      builder.reshape(var_519_scale, [1, 1, 1, 1]),
      builder.reshape(var_474_zero_point, [1, 1, 1, 1])
    );
    
    const var_523_dequantized = builder.dequantizeLinear(
      var_523_quantized,
      builder.reshape(var_523_scale, [1, 1, 1, 1]),
      builder.reshape(var_475_zero_point, [1, 1, 1, 1])
    );
    
    const var_524_dequantized = builder.dequantizeLinear(
      var_524_quantized,
      var_524_scale,
      var_623_zero_point
    );
    
    const var_522_quantizeinput = builder.conv2d(
      var_519_conv_29_dequantized, var_523_dequantized,
      {
        strides: [1, 1],
        padding: [1, 1, 1, 1],
        dilations: [1, 1],
        groups: 192,
        bias: var_524_dequantized,
        filterLayout: 'oihw',
        inputLayout: 'nchw'
      }
    );
    
    const var_522_quantized = builder.quantizeLinear(
      var_522_quantizeinput,
      builder.reshape(var_522_scale, [1, 1, 1, 1]),
      builder.reshape(var_474_zero_point, [1, 1, 1, 1])
    );
    
    const var_522_conv_31_dequantized = builder.dequantizeLinear(
      var_522_quantized,
      builder.reshape(var_522_scale, [1, 1, 1, 1]),
      builder.reshape(var_474_zero_point, [1, 1, 1, 1])
    );
    
    const var_526_dequantized = builder.dequantizeLinear(
      var_526_quantized,
      builder.reshape(var_526_scale, [1, 1, 1, 1]),
      builder.reshape(var_475_zero_point, [1, 1, 1, 1])
    );
    
    const var_527_dequantized = builder.dequantizeLinear(
      var_527_quantized,
      var_527_scale,
      var_623_zero_point
    );
    
    const var_525_quantizeinput = builder.conv2d(
      var_522_conv_31_dequantized, var_526_dequantized,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_527_dequantized,
        filterLayout: 'oihw',
        inputLayout: 'nchw'
      }
    );
    
    const var_525_quantized = builder.quantizeLinear(
      var_525_quantizeinput,
      builder.reshape(var_525_scale, [1, 1, 1, 1]),
      builder.reshape(var_525_zero_point, [1, 1, 1, 1])
    );
    
    const var_525 = builder.dequantizeLinear(
      var_525_quantized,
      builder.reshape(var_525_scale, [1, 1, 1, 1]),
      builder.reshape(var_525_zero_point, [1, 1, 1, 1])
    );
    
    const var_365 = builder.add(
      var_356,
      var_525
    );
    
    const var_365_conv_33_quantizelinear = builder.quantizeLinear(
      var_365,
      builder.reshape(var_365_scale, [1, 1, 1, 1]),
      builder.reshape(var_365_zero_point, [1, 1, 1, 1])
    );
    
    const var_365_conv_33_dequantized = builder.dequantizeLinear(
      var_365_conv_33_quantizelinear,
      builder.reshape(var_365_scale, [1, 1, 1, 1]),
      builder.reshape(var_365_zero_point, [1, 1, 1, 1])
    );
    
    const var_529_dequantized = builder.dequantizeLinear(
      var_529_quantized,
      builder.reshape(var_529_scale, [1, 1, 1, 1]),
      builder.reshape(var_475_zero_point, [1, 1, 1, 1])
    );
    
    const var_530_dequantized = builder.dequantizeLinear(
      var_530_quantized,
      var_530_scale,
      var_623_zero_point
    );
    
    const var_528_quantizeinput = builder.conv2d(
      var_365_conv_33_dequantized, var_529_dequantized,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_530_dequantized,
        filterLayout: 'oihw',
        inputLayout: 'nchw'
      }
    );
    
    const var_528_quantized = builder.quantizeLinear(
      var_528_quantizeinput,
      builder.reshape(var_528_scale, [1, 1, 1, 1]),
      builder.reshape(var_474_zero_point, [1, 1, 1, 1])
    );
    
    const var_528_conv_35_dequantized = builder.dequantizeLinear(
      var_528_quantized,
      builder.reshape(var_528_scale, [1, 1, 1, 1]),
      builder.reshape(var_474_zero_point, [1, 1, 1, 1])
    );
    
    const var_532_dequantized = builder.dequantizeLinear(
      var_532_quantized,
      builder.reshape(var_532_scale, [1, 1, 1, 1]),
      builder.reshape(var_475_zero_point, [1, 1, 1, 1])
    );
    
    const var_533_dequantized = builder.dequantizeLinear(
      var_533_quantized,
      var_533_scale,
      var_623_zero_point
    );
    
    const var_531_quantizeinput = builder.conv2d(
      var_528_conv_35_dequantized, var_532_dequantized,
      {
        strides: [2, 2],
        padding: [1, 1, 1, 1],
        dilations: [1, 1],
        groups: 192,
        bias: var_533_dequantized,
        filterLayout: 'oihw',
        inputLayout: 'nchw'
      }
    );
    
    const var_531_quantized = builder.quantizeLinear(
      var_531_quantizeinput,
      builder.reshape(var_531_scale, [1, 1, 1, 1]),
      builder.reshape(var_474_zero_point, [1, 1, 1, 1])
    );
    
    const var_531_conv_37_dequantized = builder.dequantizeLinear(
      var_531_quantized,
      builder.reshape(var_531_scale, [1, 1, 1, 1]),
      builder.reshape(var_474_zero_point, [1, 1, 1, 1])
    );
    
    const var_535_dequantized = builder.dequantizeLinear(
      var_535_quantized,
      builder.reshape(var_535_scale, [1, 1, 1, 1]),
      builder.reshape(var_475_zero_point, [1, 1, 1, 1])
    );
    
    const var_536_dequantized = builder.dequantizeLinear(
      var_536_quantized,
      var_536_scale,
      var_623_zero_point
    );
    
    const var_534_quantizeinput = builder.conv2d(
      var_531_conv_37_dequantized, var_535_dequantized,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_536_dequantized,
        filterLayout: 'oihw',
        inputLayout: 'nchw'
      }
    );
    
    const var_534_quantized = builder.quantizeLinear(
      var_534_quantizeinput,
      builder.reshape(var_534_scale, [1, 1, 1, 1]),
      builder.reshape(var_534_zero_point, [1, 1, 1, 1])
    );
    
    const var_534_duplicated = builder.dequantizeLinear(
      var_534_quantized,
      builder.reshape(var_534_scale, [1, 1, 1, 1]),
      builder.reshape(var_534_zero_point, [1, 1, 1, 1])
    );
    
    const var_538_dequantized = builder.dequantizeLinear(
      var_538_quantized,
      builder.reshape(var_538_scale, [1, 1, 1, 1]),
      builder.reshape(var_475_zero_point, [1, 1, 1, 1])
    );
    
    const var_539_dequantized = builder.dequantizeLinear(
      var_539_quantized,
      var_539_scale,
      var_623_zero_point
    );
    
    const var_537_quantizeinput = builder.conv2d(
      var_534_duplicated, var_538_dequantized,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_539_dequantized,
        filterLayout: 'oihw',
        inputLayout: 'nchw'
      }
    );
    
    const var_537_quantized = builder.quantizeLinear(
      var_537_quantizeinput,
      builder.reshape(var_537_scale, [1, 1, 1, 1]),
      builder.reshape(var_474_zero_point, [1, 1, 1, 1])
    );
    
    const var_537_conv_40_dequantized = builder.dequantizeLinear(
      var_537_quantized,
      builder.reshape(var_537_scale, [1, 1, 1, 1]),
      builder.reshape(var_474_zero_point, [1, 1, 1, 1])
    );
    
    const var_541_dequantized = builder.dequantizeLinear(
      var_541_quantized,
      builder.reshape(var_541_scale, [1, 1, 1, 1]),
      builder.reshape(var_475_zero_point, [1, 1, 1, 1])
    );
    
    const var_542_dequantized = builder.dequantizeLinear(
      var_542_quantized,
      var_542_scale,
      var_623_zero_point
    );
    
    const var_540_quantizeinput = builder.conv2d(
      var_537_conv_40_dequantized, var_541_dequantized,
      {
        strides: [1, 1],
        padding: [1, 1, 1, 1],
        dilations: [1, 1],
        groups: 384,
        bias: var_542_dequantized,
        filterLayout: 'oihw',
        inputLayout: 'nchw'
      }
    );
    
    const var_540_quantized = builder.quantizeLinear(
      var_540_quantizeinput,
      builder.reshape(var_540_scale, [1, 1, 1, 1]),
      builder.reshape(var_474_zero_point, [1, 1, 1, 1])
    );
    
    const var_540_conv_42_dequantized = builder.dequantizeLinear(
      var_540_quantized,
      builder.reshape(var_540_scale, [1, 1, 1, 1]),
      builder.reshape(var_474_zero_point, [1, 1, 1, 1])
    );
    
    const var_544_dequantized = builder.dequantizeLinear(
      var_544_quantized,
      builder.reshape(var_544_scale, [1, 1, 1, 1]),
      builder.reshape(var_475_zero_point, [1, 1, 1, 1])
    );
    
    const var_545_dequantized = builder.dequantizeLinear(
      var_545_quantized,
      var_545_scale,
      var_623_zero_point
    );
    
    const var_543_quantizeinput = builder.conv2d(
      var_540_conv_42_dequantized, var_544_dequantized,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_545_dequantized,
        filterLayout: 'oihw',
        inputLayout: 'nchw'
      }
    );
    
    const var_543_quantized = builder.quantizeLinear(
      var_543_quantizeinput,
      builder.reshape(var_543_scale, [1, 1, 1, 1]),
      builder.reshape(var_543_zero_point, [1, 1, 1, 1])
    );
    
    const var_543 = builder.dequantizeLinear(
      var_543_quantized,
      builder.reshape(var_543_scale, [1, 1, 1, 1]),
      builder.reshape(var_543_zero_point, [1, 1, 1, 1])
    );
    
    const var_382 = builder.add(
      var_534_duplicated,
      var_543
    );
    
    const var_382_conv_44_quantizelinear = builder.quantizeLinear(
      var_382,
      builder.reshape(var_382_scale, [1, 1, 1, 1]),
      builder.reshape(var_382_zero_point, [1, 1, 1, 1])
    );
    
    const var_382_conv_44_dequantized = builder.dequantizeLinear(
      var_382_conv_44_quantizelinear,
      builder.reshape(var_382_scale, [1, 1, 1, 1]),
      builder.reshape(var_382_zero_point, [1, 1, 1, 1])
    );
    
    const var_547_dequantized = builder.dequantizeLinear(
      var_547_quantized,
      builder.reshape(var_547_scale, [1, 1, 1, 1]),
      builder.reshape(var_475_zero_point, [1, 1, 1, 1])
    );
    
    const var_548_dequantized = builder.dequantizeLinear(
      var_548_quantized,
      var_548_scale,
      var_623_zero_point
    );
    
    const var_546_quantizeinput = builder.conv2d(
      var_382_conv_44_dequantized, var_547_dequantized,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_548_dequantized,
        filterLayout: 'oihw',
        inputLayout: 'nchw'
      }
    );
    
    const var_546_quantized = builder.quantizeLinear(
      var_546_quantizeinput,
      builder.reshape(var_546_scale, [1, 1, 1, 1]),
      builder.reshape(var_474_zero_point, [1, 1, 1, 1])
    );
    
    const var_546_conv_46_dequantized = builder.dequantizeLinear(
      var_546_quantized,
      builder.reshape(var_546_scale, [1, 1, 1, 1]),
      builder.reshape(var_474_zero_point, [1, 1, 1, 1])
    );
    
    const var_550_dequantized = builder.dequantizeLinear(
      var_550_quantized,
      builder.reshape(var_550_scale, [1, 1, 1, 1]),
      builder.reshape(var_475_zero_point, [1, 1, 1, 1])
    );
    
    const var_551_dequantized = builder.dequantizeLinear(
      var_551_quantized,
      var_551_scale,
      var_623_zero_point
    );
    
    const var_549_quantizeinput = builder.conv2d(
      var_546_conv_46_dequantized, var_550_dequantized,
      {
        strides: [1, 1],
        padding: [1, 1, 1, 1],
        dilations: [1, 1],
        groups: 384,
        bias: var_551_dequantized,
        filterLayout: 'oihw',
        inputLayout: 'nchw'
      }
    );
    
    const var_549_quantized = builder.quantizeLinear(
      var_549_quantizeinput,
      builder.reshape(var_549_scale, [1, 1, 1, 1]),
      builder.reshape(var_474_zero_point, [1, 1, 1, 1])
    );
    
    const var_549_conv_48_dequantized = builder.dequantizeLinear(
      var_549_quantized,
      builder.reshape(var_549_scale, [1, 1, 1, 1]),
      builder.reshape(var_474_zero_point, [1, 1, 1, 1])
    );
    
    const var_553_dequantized = builder.dequantizeLinear(
      var_553_quantized,
      builder.reshape(var_553_scale, [1, 1, 1, 1]),
      builder.reshape(var_475_zero_point, [1, 1, 1, 1])
    );
    
    const var_554_dequantized = builder.dequantizeLinear(
      var_554_quantized,
      var_554_scale,
      var_623_zero_point
    );
    
    const var_552_quantizeinput = builder.conv2d(
      var_549_conv_48_dequantized, var_553_dequantized,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_554_dequantized,
        filterLayout: 'oihw',
        inputLayout: 'nchw'
      }
    );
    
    const var_552_quantized = builder.quantizeLinear(
      var_552_quantizeinput,
      builder.reshape(var_552_scale, [1, 1, 1, 1]),
      builder.reshape(var_543_zero_point, [1, 1, 1, 1])
    );
    
    const var_552 = builder.dequantizeLinear(
      var_552_quantized,
      builder.reshape(var_552_scale, [1, 1, 1, 1]),
      builder.reshape(var_543_zero_point, [1, 1, 1, 1])
    );
    
    const var_391 = builder.add(
      var_382,
      var_552
    );
    
    const var_391_conv_50_quantizelinear = builder.quantizeLinear(
      var_391,
      builder.reshape(var_391_scale, [1, 1, 1, 1]),
      builder.reshape(var_525_zero_point, [1, 1, 1, 1])
    );
    
    const var_391_conv_50_dequantized = builder.dequantizeLinear(
      var_391_conv_50_quantizelinear,
      builder.reshape(var_391_scale, [1, 1, 1, 1]),
      builder.reshape(var_525_zero_point, [1, 1, 1, 1])
    );
    
    const var_556_dequantized = builder.dequantizeLinear(
      var_556_quantized,
      builder.reshape(var_556_scale, [1, 1, 1, 1]),
      builder.reshape(var_475_zero_point, [1, 1, 1, 1])
    );
    
    const var_557_dequantized = builder.dequantizeLinear(
      var_557_quantized,
      var_557_scale,
      var_623_zero_point
    );
    
    const var_555_quantizeinput = builder.conv2d(
      var_391_conv_50_dequantized, var_556_dequantized,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_557_dequantized,
        filterLayout: 'oihw',
        inputLayout: 'nchw'
      }
    );
    
    const var_555_quantized = builder.quantizeLinear(
      var_555_quantizeinput,
      builder.reshape(var_555_scale, [1, 1, 1, 1]),
      builder.reshape(var_474_zero_point, [1, 1, 1, 1])
    );
    
    const var_555_conv_52_dequantized = builder.dequantizeLinear(
      var_555_quantized,
      builder.reshape(var_555_scale, [1, 1, 1, 1]),
      builder.reshape(var_474_zero_point, [1, 1, 1, 1])
    );
    
    const var_559_dequantized = builder.dequantizeLinear(
      var_559_quantized,
      builder.reshape(var_559_scale, [1, 1, 1, 1]),
      builder.reshape(var_475_zero_point, [1, 1, 1, 1])
    );
    
    const var_560_dequantized = builder.dequantizeLinear(
      var_560_quantized,
      var_560_scale,
      var_623_zero_point
    );
    
    const var_558_quantizeinput = builder.conv2d(
      var_555_conv_52_dequantized, var_559_dequantized,
      {
        strides: [1, 1],
        padding: [1, 1, 1, 1],
        dilations: [1, 1],
        groups: 384,
        bias: var_560_dequantized,
        filterLayout: 'oihw',
        inputLayout: 'nchw'
      }
    );
    
    const var_558_quantized = builder.quantizeLinear(
      var_558_quantizeinput,
      builder.reshape(var_558_scale, [1, 1, 1, 1]),
      builder.reshape(var_474_zero_point, [1, 1, 1, 1])
    );
    
    const var_558_conv_54_dequantized = builder.dequantizeLinear(
      var_558_quantized,
      builder.reshape(var_558_scale, [1, 1, 1, 1]),
      builder.reshape(var_474_zero_point, [1, 1, 1, 1])
    );
    
    const var_562_dequantized = builder.dequantizeLinear(
      var_562_quantized,
      builder.reshape(var_562_scale, [1, 1, 1, 1]),
      builder.reshape(var_475_zero_point, [1, 1, 1, 1])
    );
    
    const var_563_dequantized = builder.dequantizeLinear(
      var_563_quantized,
      var_563_scale,
      var_623_zero_point
    );
    
    const var_561_quantizeinput = builder.conv2d(
      var_558_conv_54_dequantized, var_562_dequantized,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_563_dequantized,
        filterLayout: 'oihw',
        inputLayout: 'nchw'
      }
    );
    
    const var_561_quantized = builder.quantizeLinear(
      var_561_quantizeinput,
      builder.reshape(var_561_scale, [1, 1, 1, 1]),
      builder.reshape(var_561_zero_point, [1, 1, 1, 1])
    );
    
    const var_561 = builder.dequantizeLinear(
      var_561_quantized,
      builder.reshape(var_561_scale, [1, 1, 1, 1]),
      builder.reshape(var_561_zero_point, [1, 1, 1, 1])
    );
    
    const var_400 = builder.add(
      var_391,
      var_561
    );
    
    const var_400_conv_56_quantizelinear = builder.quantizeLinear(
      var_400,
      builder.reshape(var_400_scale, [1, 1, 1, 1]),
      builder.reshape(var_400_zero_point, [1, 1, 1, 1])
    );
    
    const var_400_conv_56_dequantized = builder.dequantizeLinear(
      var_400_conv_56_quantizelinear,
      builder.reshape(var_400_scale, [1, 1, 1, 1]),
      builder.reshape(var_400_zero_point, [1, 1, 1, 1])
    );
    
    const var_565_dequantized = builder.dequantizeLinear(
      var_565_quantized,
      builder.reshape(var_565_scale, [1, 1, 1, 1]),
      builder.reshape(var_475_zero_point, [1, 1, 1, 1])
    );
    
    const var_566_dequantized = builder.dequantizeLinear(
      var_566_quantized,
      var_566_scale,
      var_623_zero_point
    );
    
    const var_564_quantizeinput = builder.conv2d(
      var_400_conv_56_dequantized, var_565_dequantized,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_566_dequantized,
        filterLayout: 'oihw',
        inputLayout: 'nchw'
      }
    );
    
    const var_564_quantized = builder.quantizeLinear(
      var_564_quantizeinput,
      builder.reshape(var_564_scale, [1, 1, 1, 1]),
      builder.reshape(var_474_zero_point, [1, 1, 1, 1])
    );
    
    const var_564_conv_58_dequantized = builder.dequantizeLinear(
      var_564_quantized,
      builder.reshape(var_564_scale, [1, 1, 1, 1]),
      builder.reshape(var_474_zero_point, [1, 1, 1, 1])
    );
    
    const var_568_dequantized = builder.dequantizeLinear(
      var_568_quantized,
      builder.reshape(var_568_scale, [1, 1, 1, 1]),
      builder.reshape(var_475_zero_point, [1, 1, 1, 1])
    );
    
    const var_569_dequantized = builder.dequantizeLinear(
      var_569_quantized,
      var_569_scale,
      var_623_zero_point
    );
    
    const var_567_quantizeinput = builder.conv2d(
      var_564_conv_58_dequantized, var_568_dequantized,
      {
        strides: [1, 1],
        padding: [1, 1, 1, 1],
        dilations: [1, 1],
        groups: 384,
        bias: var_569_dequantized,
        filterLayout: 'oihw',
        inputLayout: 'nchw'
      }
    );
    
    const var_567_quantized = builder.quantizeLinear(
      var_567_quantizeinput,
      builder.reshape(var_567_scale, [1, 1, 1, 1]),
      builder.reshape(var_474_zero_point, [1, 1, 1, 1])
    );
    
    const var_567_conv_60_dequantized = builder.dequantizeLinear(
      var_567_quantized,
      builder.reshape(var_567_scale, [1, 1, 1, 1]),
      builder.reshape(var_474_zero_point, [1, 1, 1, 1])
    );
    
    const var_571_dequantized = builder.dequantizeLinear(
      var_571_quantized,
      builder.reshape(var_571_scale, [1, 1, 1, 1]),
      builder.reshape(var_475_zero_point, [1, 1, 1, 1])
    );
    
    const var_572_dequantized = builder.dequantizeLinear(
      var_572_quantized,
      var_572_scale,
      var_623_zero_point
    );
    
    const var_570_quantizeinput = builder.conv2d(
      var_567_conv_60_dequantized, var_571_dequantized,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_572_dequantized,
        filterLayout: 'oihw',
        inputLayout: 'nchw'
      }
    );
    
    const var_570_quantized = builder.quantizeLinear(
      var_570_quantizeinput,
      builder.reshape(var_570_scale, [1, 1, 1, 1]),
      builder.reshape(var_624_zero_point, [1, 1, 1, 1])
    );
    
    const var_570_duplicated = builder.dequantizeLinear(
      var_570_quantized,
      builder.reshape(var_570_scale, [1, 1, 1, 1]),
      builder.reshape(var_624_zero_point, [1, 1, 1, 1])
    );
    
    const var_574_dequantized = builder.dequantizeLinear(
      var_574_quantized,
      builder.reshape(var_574_scale, [1, 1, 1, 1]),
      builder.reshape(var_475_zero_point, [1, 1, 1, 1])
    );
    
    const var_575_dequantized = builder.dequantizeLinear(
      var_575_quantized,
      var_575_scale,
      var_623_zero_point
    );
    
    const var_573_quantizeinput = builder.conv2d(
      var_570_duplicated, var_574_dequantized,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_575_dequantized,
        filterLayout: 'oihw',
        inputLayout: 'nchw'
      }
    );
    
    const var_573_quantized = builder.quantizeLinear(
      var_573_quantizeinput,
      builder.reshape(var_573_scale, [1, 1, 1, 1]),
      builder.reshape(var_474_zero_point, [1, 1, 1, 1])
    );
    
    const var_573_conv_63_dequantized = builder.dequantizeLinear(
      var_573_quantized,
      builder.reshape(var_573_scale, [1, 1, 1, 1]),
      builder.reshape(var_474_zero_point, [1, 1, 1, 1])
    );
    
    const var_577_dequantized = builder.dequantizeLinear(
      var_577_quantized,
      builder.reshape(var_577_scale, [1, 1, 1, 1]),
      builder.reshape(var_475_zero_point, [1, 1, 1, 1])
    );
    
    const var_578_dequantized = builder.dequantizeLinear(
      var_578_quantized,
      var_578_scale,
      var_623_zero_point
    );
    
    const var_576_quantizeinput = builder.conv2d(
      var_573_conv_63_dequantized, var_577_dequantized,
      {
        strides: [1, 1],
        padding: [1, 1, 1, 1],
        dilations: [1, 1],
        groups: 576,
        bias: var_578_dequantized,
        filterLayout: 'oihw',
        inputLayout: 'nchw'
      }
    );
    
    const var_576_quantized = builder.quantizeLinear(
      var_576_quantizeinput,
      builder.reshape(var_576_scale, [1, 1, 1, 1]),
      builder.reshape(var_474_zero_point, [1, 1, 1, 1])
    );
    
    const var_576_conv_65_dequantized = builder.dequantizeLinear(
      var_576_quantized,
      builder.reshape(var_576_scale, [1, 1, 1, 1]),
      builder.reshape(var_474_zero_point, [1, 1, 1, 1])
    );
    
    const var_580_dequantized = builder.dequantizeLinear(
      var_580_quantized,
      builder.reshape(var_580_scale, [1, 1, 1, 1]),
      builder.reshape(var_475_zero_point, [1, 1, 1, 1])
    );
    
    const var_581_dequantized = builder.dequantizeLinear(
      var_581_quantized,
      var_581_scale,
      var_623_zero_point
    );
    
    const var_579_quantizeinput = builder.conv2d(
      var_576_conv_65_dequantized, var_580_dequantized,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_581_dequantized,
        filterLayout: 'oihw',
        inputLayout: 'nchw'
      }
    );
    
    const var_579_quantized = builder.quantizeLinear(
      var_579_quantizeinput,
      builder.reshape(var_579_scale, [1, 1, 1, 1]),
      builder.reshape(var_579_zero_point, [1, 1, 1, 1])
    );
    
    const var_579 = builder.dequantizeLinear(
      var_579_quantized,
      builder.reshape(var_579_scale, [1, 1, 1, 1]),
      builder.reshape(var_579_zero_point, [1, 1, 1, 1])
    );
    
    const var_417 = builder.add(
      var_570_duplicated,
      var_579
    );
    
    const var_417_conv_67_quantizelinear = builder.quantizeLinear(
      var_417,
      builder.reshape(var_417_scale, [1, 1, 1, 1]),
      builder.reshape(var_417_zero_point, [1, 1, 1, 1])
    );
    
    const var_417_conv_67_dequantized = builder.dequantizeLinear(
      var_417_conv_67_quantizelinear,
      builder.reshape(var_417_scale, [1, 1, 1, 1]),
      builder.reshape(var_417_zero_point, [1, 1, 1, 1])
    );
    
    const var_583_dequantized = builder.dequantizeLinear(
      var_583_quantized,
      builder.reshape(var_583_scale, [1, 1, 1, 1]),
      builder.reshape(var_475_zero_point, [1, 1, 1, 1])
    );
    
    const var_584_dequantized = builder.dequantizeLinear(
      var_584_quantized,
      var_584_scale,
      var_623_zero_point
    );
    
    const var_582_quantizeinput = builder.conv2d(
      var_417_conv_67_dequantized, var_583_dequantized,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_584_dequantized,
        filterLayout: 'oihw',
        inputLayout: 'nchw'
      }
    );
    
    const var_582_quantized = builder.quantizeLinear(
      var_582_quantizeinput,
      builder.reshape(var_582_scale, [1, 1, 1, 1]),
      builder.reshape(var_474_zero_point, [1, 1, 1, 1])
    );
    
    const var_582_conv_69_dequantized = builder.dequantizeLinear(
      var_582_quantized,
      builder.reshape(var_582_scale, [1, 1, 1, 1]),
      builder.reshape(var_474_zero_point, [1, 1, 1, 1])
    );
    
    const var_586_dequantized = builder.dequantizeLinear(
      var_586_quantized,
      builder.reshape(var_586_scale, [1, 1, 1, 1]),
      builder.reshape(var_475_zero_point, [1, 1, 1, 1])
    );
    
    const var_587_dequantized = builder.dequantizeLinear(
      var_587_quantized,
      var_587_scale,
      var_623_zero_point
    );
    
    const var_585_quantizeinput = builder.conv2d(
      var_582_conv_69_dequantized, var_586_dequantized,
      {
        strides: [1, 1],
        padding: [1, 1, 1, 1],
        dilations: [1, 1],
        groups: 576,
        bias: var_587_dequantized,
        filterLayout: 'oihw',
        inputLayout: 'nchw'
      }
    );
    
    const var_585_quantized = builder.quantizeLinear(
      var_585_quantizeinput,
      builder.reshape(var_576_scale, [1, 1, 1, 1]),
      builder.reshape(var_474_zero_point, [1, 1, 1, 1])
    );
    
    const var_585_conv_71_dequantized = builder.dequantizeLinear(
      var_585_quantized,
      builder.reshape(var_576_scale, [1, 1, 1, 1]),
      builder.reshape(var_474_zero_point, [1, 1, 1, 1])
    );
    
    const var_589_dequantized = builder.dequantizeLinear(
      var_589_quantized,
      builder.reshape(var_589_scale, [1, 1, 1, 1]),
      builder.reshape(var_475_zero_point, [1, 1, 1, 1])
    );
    
    const var_590_dequantized = builder.dequantizeLinear(
      var_590_quantized,
      var_590_scale,
      var_623_zero_point
    );
    
    const var_588_quantizeinput = builder.conv2d(
      var_585_conv_71_dequantized, var_589_dequantized,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_590_dequantized,
        filterLayout: 'oihw',
        inputLayout: 'nchw'
      }
    );
    
    const var_588_quantized = builder.quantizeLinear(
      var_588_quantizeinput,
      builder.reshape(var_588_scale, [1, 1, 1, 1]),
      builder.reshape(var_588_zero_point, [1, 1, 1, 1])
    );
    
    const var_588 = builder.dequantizeLinear(
      var_588_quantized,
      builder.reshape(var_588_scale, [1, 1, 1, 1]),
      builder.reshape(var_588_zero_point, [1, 1, 1, 1])
    );
    
    const var_426 = builder.add(
      var_417,
      var_588
    );
    
    const var_426_conv_73_quantizelinear = builder.quantizeLinear(
      var_426,
      builder.reshape(var_426_scale, [1, 1, 1, 1]),
      builder.reshape(var_624_zero_point, [1, 1, 1, 1])
    );
    
    const var_426_conv_73_dequantized = builder.dequantizeLinear(
      var_426_conv_73_quantizelinear,
      builder.reshape(var_426_scale, [1, 1, 1, 1]),
      builder.reshape(var_624_zero_point, [1, 1, 1, 1])
    );
    
    const var_592_dequantized = builder.dequantizeLinear(
      var_592_quantized,
      builder.reshape(var_592_scale, [1, 1, 1, 1]),
      builder.reshape(var_475_zero_point, [1, 1, 1, 1])
    );
    
    const var_593_dequantized = builder.dequantizeLinear(
      var_593_quantized,
      var_593_scale,
      var_623_zero_point
    );
    
    const var_591_quantizeinput = builder.conv2d(
      var_426_conv_73_dequantized, var_592_dequantized,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_593_dequantized,
        filterLayout: 'oihw',
        inputLayout: 'nchw'
      }
    );
    
    const var_591_quantized = builder.quantizeLinear(
      var_591_quantizeinput,
      builder.reshape(var_591_scale, [1, 1, 1, 1]),
      builder.reshape(var_474_zero_point, [1, 1, 1, 1])
    );
    
    const var_591_conv_75_dequantized = builder.dequantizeLinear(
      var_591_quantized,
      builder.reshape(var_591_scale, [1, 1, 1, 1]),
      builder.reshape(var_474_zero_point, [1, 1, 1, 1])
    );
    
    const var_595_dequantized = builder.dequantizeLinear(
      var_595_quantized,
      builder.reshape(var_595_scale, [1, 1, 1, 1]),
      builder.reshape(var_475_zero_point, [1, 1, 1, 1])
    );
    
    const var_596_dequantized = builder.dequantizeLinear(
      var_596_quantized,
      var_596_scale,
      var_623_zero_point
    );
    
    const var_594_quantizeinput = builder.conv2d(
      var_591_conv_75_dequantized, var_595_dequantized,
      {
        strides: [2, 2],
        padding: [1, 1, 1, 1],
        dilations: [1, 1],
        groups: 576,
        bias: var_596_dequantized,
        filterLayout: 'oihw',
        inputLayout: 'nchw'
      }
    );
    
    const var_594_quantized = builder.quantizeLinear(
      var_594_quantizeinput,
      builder.reshape(var_576_scale, [1, 1, 1, 1]),
      builder.reshape(var_474_zero_point, [1, 1, 1, 1])
    );
    
    const var_594_conv_77_dequantized = builder.dequantizeLinear(
      var_594_quantized,
      builder.reshape(var_576_scale, [1, 1, 1, 1]),
      builder.reshape(var_474_zero_point, [1, 1, 1, 1])
    );
    
    const var_598_dequantized = builder.dequantizeLinear(
      var_598_quantized,
      builder.reshape(var_598_scale, [1, 1, 1, 1]),
      builder.reshape(var_475_zero_point, [1, 1, 1, 1])
    );
    
    const var_599_dequantized = builder.dequantizeLinear(
      var_599_quantized,
      var_599_scale,
      var_623_zero_point
    );
    
    const var_597_quantizeinput = builder.conv2d(
      var_594_conv_77_dequantized, var_598_dequantized,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_599_dequantized,
        filterLayout: 'oihw',
        inputLayout: 'nchw'
      }
    );
    
    const var_597_quantized = builder.quantizeLinear(
      var_597_quantizeinput,
      builder.reshape(var_597_scale, [1, 1, 1, 1]),
      builder.reshape(var_597_zero_point, [1, 1, 1, 1])
    );
    
    const var_597_duplicated = builder.dequantizeLinear(
      var_597_quantized,
      builder.reshape(var_597_scale, [1, 1, 1, 1]),
      builder.reshape(var_597_zero_point, [1, 1, 1, 1])
    );
    
    const var_601_dequantized = builder.dequantizeLinear(
      var_601_quantized,
      builder.reshape(var_601_scale, [1, 1, 1, 1]),
      builder.reshape(var_475_zero_point, [1, 1, 1, 1])
    );
    
    const var_602_dequantized = builder.dequantizeLinear(
      var_602_quantized,
      var_602_scale,
      var_623_zero_point
    );
    
    const var_600_quantizeinput = builder.conv2d(
      var_597_duplicated, var_601_dequantized,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_602_dequantized,
        filterLayout: 'oihw',
        inputLayout: 'nchw'
      }
    );
    
    const var_600_quantized = builder.quantizeLinear(
      var_600_quantizeinput,
      builder.reshape(var_600_scale, [1, 1, 1, 1]),
      builder.reshape(var_474_zero_point, [1, 1, 1, 1])
    );
    
    const var_600_conv_80_dequantized = builder.dequantizeLinear(
      var_600_quantized,
      builder.reshape(var_600_scale, [1, 1, 1, 1]),
      builder.reshape(var_474_zero_point, [1, 1, 1, 1])
    );
    
    const var_604_dequantized = builder.dequantizeLinear(
      var_604_quantized,
      builder.reshape(var_604_scale, [1, 1, 1, 1]),
      builder.reshape(var_475_zero_point, [1, 1, 1, 1])
    );
    
    const var_605_dequantized = builder.dequantizeLinear(
      var_605_quantized,
      var_605_scale,
      var_623_zero_point
    );
    
    const var_603_quantizeinput = builder.conv2d(
      var_600_conv_80_dequantized, var_604_dequantized,
      {
        strides: [1, 1],
        padding: [1, 1, 1, 1],
        dilations: [1, 1],
        groups: 960,
        bias: var_605_dequantized,
        filterLayout: 'oihw',
        inputLayout: 'nchw'
      }
    );
    
    const var_603_quantized = builder.quantizeLinear(
      var_603_quantizeinput,
      builder.reshape(var_603_scale, [1, 1, 1, 1]),
      builder.reshape(var_474_zero_point, [1, 1, 1, 1])
    );
    
    const var_603_conv_82_dequantized = builder.dequantizeLinear(
      var_603_quantized,
      builder.reshape(var_603_scale, [1, 1, 1, 1]),
      builder.reshape(var_474_zero_point, [1, 1, 1, 1])
    );
    
    const var_607_dequantized = builder.dequantizeLinear(
      var_607_quantized,
      builder.reshape(var_607_scale, [1, 1, 1, 1]),
      builder.reshape(var_475_zero_point, [1, 1, 1, 1])
    );
    
    const var_608_dequantized = builder.dequantizeLinear(
      var_608_quantized,
      var_608_scale,
      var_623_zero_point
    );
    
    const var_606_quantizeinput = builder.conv2d(
      var_603_conv_82_dequantized, var_607_dequantized,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_608_dequantized,
        filterLayout: 'oihw',
        inputLayout: 'nchw'
      }
    );
    
    const var_606_quantized = builder.quantizeLinear(
      var_606_quantizeinput,
      builder.reshape(var_606_scale, [1, 1, 1, 1]),
      builder.reshape(var_606_zero_point, [1, 1, 1, 1])
    );
    
    const var_606 = builder.dequantizeLinear(
      var_606_quantized,
      builder.reshape(var_606_scale, [1, 1, 1, 1]),
      builder.reshape(var_606_zero_point, [1, 1, 1, 1])
    );
    
    const var_443 = builder.add(
      var_597_duplicated,
      var_606
    );
    
    const var_443_conv_84_quantizelinear = builder.quantizeLinear(
      var_443,
      builder.reshape(var_443_scale, [1, 1, 1, 1]),
      builder.reshape(var_480_zero_point, [1, 1, 1, 1])
    );
    
    const var_443_conv_84_dequantized = builder.dequantizeLinear(
      var_443_conv_84_quantizelinear,
      builder.reshape(var_443_scale, [1, 1, 1, 1]),
      builder.reshape(var_480_zero_point, [1, 1, 1, 1])
    );
    
    const var_610_dequantized = builder.dequantizeLinear(
      var_610_quantized,
      builder.reshape(var_610_scale, [1, 1, 1, 1]),
      builder.reshape(var_475_zero_point, [1, 1, 1, 1])
    );
    
    const var_611_dequantized = builder.dequantizeLinear(
      var_611_quantized,
      var_611_scale,
      var_623_zero_point
    );
    
    const var_609_quantizeinput = builder.conv2d(
      var_443_conv_84_dequantized, var_610_dequantized,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_611_dequantized,
        filterLayout: 'oihw',
        inputLayout: 'nchw'
      }
    );
    
    const var_609_quantized = builder.quantizeLinear(
      var_609_quantizeinput,
      builder.reshape(var_609_scale, [1, 1, 1, 1]),
      builder.reshape(var_474_zero_point, [1, 1, 1, 1])
    );
    
    const var_609_conv_86_dequantized = builder.dequantizeLinear(
      var_609_quantized,
      builder.reshape(var_609_scale, [1, 1, 1, 1]),
      builder.reshape(var_474_zero_point, [1, 1, 1, 1])
    );
    
    const var_613_dequantized = builder.dequantizeLinear(
      var_613_quantized,
      builder.reshape(var_613_scale, [1, 1, 1, 1]),
      builder.reshape(var_475_zero_point, [1, 1, 1, 1])
    );
    
    const var_614_dequantized = builder.dequantizeLinear(
      var_614_quantized,
      var_614_scale,
      var_623_zero_point
    );
    
    const var_612_quantizeinput = builder.conv2d(
      var_609_conv_86_dequantized, var_613_dequantized,
      {
        strides: [1, 1],
        padding: [1, 1, 1, 1],
        dilations: [1, 1],
        groups: 960,
        bias: var_614_dequantized,
        filterLayout: 'oihw',
        inputLayout: 'nchw'
      }
    );
    
    const var_612_quantized = builder.quantizeLinear(
      var_612_quantizeinput,
      builder.reshape(var_576_scale, [1, 1, 1, 1]),
      builder.reshape(var_474_zero_point, [1, 1, 1, 1])
    );
    
    const var_612_conv_88_dequantized = builder.dequantizeLinear(
      var_612_quantized,
      builder.reshape(var_576_scale, [1, 1, 1, 1]),
      builder.reshape(var_474_zero_point, [1, 1, 1, 1])
    );
    
    const var_616_dequantized = builder.dequantizeLinear(
      var_616_quantized,
      builder.reshape(var_616_scale, [1, 1, 1, 1]),
      builder.reshape(var_475_zero_point, [1, 1, 1, 1])
    );
    
    const var_617_dequantized = builder.dequantizeLinear(
      var_617_quantized,
      var_617_scale,
      var_623_zero_point
    );
    
    const var_615_quantizeinput = builder.conv2d(
      var_612_conv_88_dequantized, var_616_dequantized,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_617_dequantized,
        filterLayout: 'oihw',
        inputLayout: 'nchw'
      }
    );
    
    const var_615_quantized = builder.quantizeLinear(
      var_615_quantizeinput,
      builder.reshape(var_615_scale, [1, 1, 1, 1]),
      builder.reshape(var_615_zero_point, [1, 1, 1, 1])
    );
    
    const var_615 = builder.dequantizeLinear(
      var_615_quantized,
      builder.reshape(var_615_scale, [1, 1, 1, 1]),
      builder.reshape(var_615_zero_point, [1, 1, 1, 1])
    );
    
    const var_452 = builder.add(
      var_443,
      var_615
    );
    
    const var_452_conv_90_quantizelinear = builder.quantizeLinear(
      var_452,
      builder.reshape(var_452_scale, [1, 1, 1, 1]),
      builder.reshape(var_452_zero_point, [1, 1, 1, 1])
    );
    
    const var_452_conv_90_dequantized = builder.dequantizeLinear(
      var_452_conv_90_quantizelinear,
      builder.reshape(var_452_scale, [1, 1, 1, 1]),
      builder.reshape(var_452_zero_point, [1, 1, 1, 1])
    );
    
    const var_619_dequantized = builder.dequantizeLinear(
      var_619_quantized,
      builder.reshape(var_619_scale, [1, 1, 1, 1]),
      builder.reshape(var_475_zero_point, [1, 1, 1, 1])
    );
    
    const var_620_dequantized = builder.dequantizeLinear(
      var_620_quantized,
      var_620_scale,
      var_623_zero_point
    );
    
    const var_618_quantizeinput = builder.conv2d(
      var_452_conv_90_dequantized, var_619_dequantized,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_620_dequantized,
        filterLayout: 'oihw',
        inputLayout: 'nchw'
      }
    );
    
    const var_618_quantized = builder.quantizeLinear(
      var_618_quantizeinput,
      builder.reshape(var_618_scale, [1, 1, 1, 1]),
      builder.reshape(var_474_zero_point, [1, 1, 1, 1])
    );
    
    const var_618_conv_92_dequantized = builder.dequantizeLinear(
      var_618_quantized,
      builder.reshape(var_618_scale, [1, 1, 1, 1]),
      builder.reshape(var_474_zero_point, [1, 1, 1, 1])
    );
    
    const var_622_dequantized = builder.dequantizeLinear(
      var_622_quantized,
      builder.reshape(var_622_scale, [1, 1, 1, 1]),
      builder.reshape(var_475_zero_point, [1, 1, 1, 1])
    );
    
    const var_623_dequantized = builder.dequantizeLinear(
      var_623_quantized,
      var_623_scale,
      var_623_zero_point
    );
    
    const var_621_quantizeinput = builder.conv2d(
      var_618_conv_92_dequantized, var_622_dequantized,
      {
        strides: [1, 1],
        padding: [1, 1, 1, 1],
        dilations: [1, 1],
        groups: 960,
        bias: var_623_dequantized,
        filterLayout: 'oihw',
        inputLayout: 'nchw'
      }
    );
    
    const var_621_quantized = builder.quantizeLinear(
      var_621_quantizeinput,
      builder.reshape(var_621_scale, [1, 1, 1, 1]),
      builder.reshape(var_474_zero_point, [1, 1, 1, 1])
    );
    
    const var_621_conv_94_dequantized = builder.dequantizeLinear(
      var_621_quantized,
      builder.reshape(var_621_scale, [1, 1, 1, 1]),
      builder.reshape(var_474_zero_point, [1, 1, 1, 1])
    );
    
    const var_625_dequantized = builder.dequantizeLinear(
      var_625_quantized,
      builder.reshape(var_625_scale, [1, 1, 1, 1]),
      builder.reshape(var_475_zero_point, [1, 1, 1, 1])
    );
    
    const var_626_dequantized = builder.dequantizeLinear(
      var_626_quantized,
      var_626_scale,
      var_623_zero_point
    );
    
    const var_624_quantizeinput = builder.conv2d(
      var_621_conv_94_dequantized, var_625_dequantized,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_626_dequantized,
        filterLayout: 'oihw',
        inputLayout: 'nchw'
      }
    );
    
    const var_624_quantized = builder.quantizeLinear(
      var_624_quantizeinput,
      builder.reshape(var_624_scale, [1, 1, 1, 1]),
      builder.reshape(var_624_zero_point, [1, 1, 1, 1])
    );
    
    const var_624_conv_95_dequantized = builder.dequantizeLinear(
      var_624_quantized,
      builder.reshape(var_624_scale, [1, 1, 1, 1]),
      builder.reshape(var_624_zero_point, [1, 1, 1, 1])
    );
    
    const var_628_dequantized = builder.dequantizeLinear(
      var_628_quantized,
      builder.reshape(var_628_scale, [1, 1, 1, 1]),
      builder.reshape(var_475_zero_point, [1, 1, 1, 1])
    );
    
    const var_629_dequantized = builder.dequantizeLinear(
      var_629_quantized,
      var_629_scale,
      var_623_zero_point
    );
    
    const var_627_quantizeinput = builder.conv2d(
      var_624_conv_95_dequantized, var_628_dequantized,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_629_dequantized,
        filterLayout: 'oihw',
        inputLayout: 'nchw'
      }
    );
    
    const var_627_quantized = builder.quantizeLinear(
      var_627_quantizeinput,
      builder.reshape(var_576_scale, [1, 1, 1, 1]),
      builder.reshape(var_474_zero_point, [1, 1, 1, 1])
    );
    
    const var_627_duplicated = builder.dequantizeLinear(
      var_627_quantized,
      builder.reshape(var_576_scale, [1, 1, 1, 1]),
      builder.reshape(var_474_zero_point, [1, 1, 1, 1])
    );
    
    const var_464_quantizeinput = builder.averagePool2d(
      var_627_duplicated
    );
    
    const var_464_quantized = builder.quantizeLinear(
      var_464_quantizeinput,
      builder.reshape(var_464_scale, [1, 1, 1, 1]),
      builder.reshape(var_474_zero_point, [1, 1, 1, 1])
    );
    
    const var_464_reshape_103_dequantized = builder.dequantizeLinear(
      var_464_quantized,
      builder.reshape(var_464_scale, [1, 1, 1, 1]),
      builder.reshape(var_474_zero_point, [1, 1, 1, 1])
    );
    
    const var_472_quantizeinput = builder.reshape(
      var_464_reshape_103_dequantized,
      (() => {
        const shape = Array.from(new BigInt64Array(weights_array_buffer.slice(2259300, 2259316)), Number);
        // Calculate the concrete size for value -1.
        if (shape.includes(-1)) {
          const count = shape.filter(v => v === -1).length;
          if (count !== 1) {
            throw new Error('Only one -1 is allowed in reshape shape');
          }
          const totalInput = var_464_reshape_103_dequantized.shape.reduce((a, b) => a * b, 1);
          const known = shape.reduce((a, b) => b === -1 ? a : a * b, 1);
          const idx = shape.indexOf(-1);
          shape[idx] = totalInput / known;
        }
        return shape;
      })()
    );
    
    const var_472_quantized = builder.quantizeLinear(
      var_472_quantizeinput,
      builder.reshape(var_464_scale, [1, 1]),
      builder.reshape(var_474_zero_point, [1, 1])
    );
    
    const var_472_gemm_104_matmul_dequantized = builder.dequantizeLinear(
      var_472_quantized,
      builder.reshape(var_464_scale, [1, 1]),
      builder.reshape(var_474_zero_point, [1, 1])
    );
    
    const var_classifier_1_weight_dequantized = builder.dequantizeLinear(
      var_classifier_1_weight_quantized,
      builder.reshape(var_classifier_1_weight_scale, [1, 1]),
      builder.reshape(var_475_zero_point, [1, 1])
    );
    
    const output_matmul_quantizeinput = builder.matmul(
      var_472_gemm_104_matmul_dequantized,
      var_classifier_1_weight_dequantized
    );
    
    const output_matmul_quantized = builder.quantizeLinear(
      output_matmul_quantizeinput,
      builder.reshape(output_matmul_scale, [1, 1]),
      builder.reshape(output_matmul_zero_point, [1, 1])
    );
    
    const output_matmul = builder.dequantizeLinear(
      output_matmul_quantized,
      builder.reshape(output_matmul_scale, [1, 1]),
      builder.reshape(output_matmul_zero_point, [1, 1])
    );
    
    const output = builder.add(
      output_matmul,
      var_classifier_1_bias
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