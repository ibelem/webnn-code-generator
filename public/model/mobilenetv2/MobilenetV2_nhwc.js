// WebNN Code Generator (NHWC)

export class MobilenetV2Nhwc {

  // Set freeDimensionOverrides globally for symbolic dimensions
  // batch_size: 1

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
      builder.input('pixel_values', { dataType: 'float32', shape: [1,3,224,224] }),
      { permutation: [0, 2, 3, 1] }
    );

    this.inputTensors_['pixel_values'] = await this.context_.createTensor(
      { dataType: 'float32', shape: [1,3,224,224], writable: true }
    );

    // Initializers, create graph constant operands
    
    const var_onnx__conv_1737 = builder.constant(
      { dataType: 'float16', shape: [32,3,3,3] },
      new Float16Array(weights_array_buffer, 0, 1728 / Float16Array.BYTES_PER_ELEMENT)
    );

    const var_onnx__conv_1738 = builder.constant(
      { dataType: 'float16', shape: [32] },
      new Float16Array(weights_array_buffer, 1728, 64 / Float16Array.BYTES_PER_ELEMENT)
    );

    const var_onnx__conv_1740 = builder.constant(
      { dataType: 'float16', shape: [1,3,3,32] },
      new Float16Array(weights_array_buffer, 1808, 576 / Float16Array.BYTES_PER_ELEMENT)
    );

    const var_onnx__conv_1741 = builder.constant(
      { dataType: 'float16', shape: [32] },
      new Float16Array(weights_array_buffer, 2384, 64 / Float16Array.BYTES_PER_ELEMENT)
    );

    const var_onnx__conv_1743 = builder.constant(
      { dataType: 'float16', shape: [16,1,1,32] },
      new Float16Array(weights_array_buffer, 2464, 1024 / Float16Array.BYTES_PER_ELEMENT)
    );

    const var_onnx__conv_1744 = builder.constant(
      { dataType: 'float16', shape: [16] },
      new Float16Array(weights_array_buffer, 3488, 32 / Float16Array.BYTES_PER_ELEMENT)
    );

    const var_onnx__conv_1746 = builder.constant(
      { dataType: 'float16', shape: [96,1,1,16] },
      new Float16Array(weights_array_buffer, 3520, 3072 / Float16Array.BYTES_PER_ELEMENT)
    );

    const var_onnx__conv_1747 = builder.constant(
      { dataType: 'float16', shape: [96] },
      new Float16Array(weights_array_buffer, 6592, 192 / Float16Array.BYTES_PER_ELEMENT)
    );

    const var_onnx__conv_1749 = builder.constant(
      { dataType: 'float16', shape: [1,3,3,96] },
      new Float16Array(weights_array_buffer, 6800, 1728 / Float16Array.BYTES_PER_ELEMENT)
    );

    const var_onnx__conv_1750 = builder.constant(
      { dataType: 'float16', shape: [96] },
      new Float16Array(weights_array_buffer, 8528, 192 / Float16Array.BYTES_PER_ELEMENT)
    );

    const var_onnx__conv_1752 = builder.constant(
      { dataType: 'float16', shape: [24,1,1,96] },
      new Float16Array(weights_array_buffer, 8736, 4608 / Float16Array.BYTES_PER_ELEMENT)
    );

    const var_onnx__conv_1753 = builder.constant(
      { dataType: 'float16', shape: [24] },
      new Float16Array(weights_array_buffer, 13344, 48 / Float16Array.BYTES_PER_ELEMENT)
    );

    const var_onnx__conv_1755 = builder.constant(
      { dataType: 'float16', shape: [144,1,1,24] },
      new Float16Array(weights_array_buffer, 13392, 6912 / Float16Array.BYTES_PER_ELEMENT)
    );

    const var_onnx__conv_1756 = builder.constant(
      { dataType: 'float16', shape: [144] },
      new Float16Array(weights_array_buffer, 20304, 288 / Float16Array.BYTES_PER_ELEMENT)
    );

    const var_onnx__conv_1758 = builder.constant(
      { dataType: 'float16', shape: [1,3,3,144] },
      new Float16Array(weights_array_buffer, 20608, 2592 / Float16Array.BYTES_PER_ELEMENT)
    );

    const var_onnx__conv_1759 = builder.constant(
      { dataType: 'float16', shape: [144] },
      new Float16Array(weights_array_buffer, 23200, 288 / Float16Array.BYTES_PER_ELEMENT)
    );

    const var_onnx__conv_1761 = builder.constant(
      { dataType: 'float16', shape: [24,1,1,144] },
      new Float16Array(weights_array_buffer, 23504, 6912 / Float16Array.BYTES_PER_ELEMENT)
    );

    const var_onnx__conv_1762 = builder.constant(
      { dataType: 'float16', shape: [24] },
      new Float16Array(weights_array_buffer, 30416, 48 / Float16Array.BYTES_PER_ELEMENT)
    );

    const var_onnx__conv_1764 = builder.constant(
      { dataType: 'float16', shape: [144,1,1,24] },
      new Float16Array(weights_array_buffer, 30464, 6912 / Float16Array.BYTES_PER_ELEMENT)
    );

    const var_onnx__conv_1765 = builder.constant(
      { dataType: 'float16', shape: [144] },
      new Float16Array(weights_array_buffer, 37376, 288 / Float16Array.BYTES_PER_ELEMENT)
    );

    const var_onnx__conv_1767 = builder.constant(
      { dataType: 'float16', shape: [1,3,3,144] },
      new Float16Array(weights_array_buffer, 37680, 2592 / Float16Array.BYTES_PER_ELEMENT)
    );

    const var_onnx__conv_1768 = builder.constant(
      { dataType: 'float16', shape: [144] },
      new Float16Array(weights_array_buffer, 40272, 288 / Float16Array.BYTES_PER_ELEMENT)
    );

    const var_onnx__conv_1770 = builder.constant(
      { dataType: 'float16', shape: [32,1,1,144] },
      new Float16Array(weights_array_buffer, 40576, 9216 / Float16Array.BYTES_PER_ELEMENT)
    );

    const var_onnx__conv_1771 = builder.constant(
      { dataType: 'float16', shape: [32] },
      new Float16Array(weights_array_buffer, 49792, 64 / Float16Array.BYTES_PER_ELEMENT)
    );

    const var_onnx__conv_1773 = builder.constant(
      { dataType: 'float16', shape: [192,1,1,32] },
      new Float16Array(weights_array_buffer, 49856, 12288 / Float16Array.BYTES_PER_ELEMENT)
    );

    const var_onnx__conv_1774 = builder.constant(
      { dataType: 'float16', shape: [192] },
      new Float16Array(weights_array_buffer, 62144, 384 / Float16Array.BYTES_PER_ELEMENT)
    );

    const var_onnx__conv_1776 = builder.constant(
      { dataType: 'float16', shape: [1,3,3,192] },
      new Float16Array(weights_array_buffer, 62544, 3456 / Float16Array.BYTES_PER_ELEMENT)
    );

    const var_onnx__conv_1777 = builder.constant(
      { dataType: 'float16', shape: [192] },
      new Float16Array(weights_array_buffer, 66000, 384 / Float16Array.BYTES_PER_ELEMENT)
    );

    const var_onnx__conv_1779 = builder.constant(
      { dataType: 'float16', shape: [32,1,1,192] },
      new Float16Array(weights_array_buffer, 66400, 12288 / Float16Array.BYTES_PER_ELEMENT)
    );

    const var_onnx__conv_1780 = builder.constant(
      { dataType: 'float16', shape: [32] },
      new Float16Array(weights_array_buffer, 78688, 64 / Float16Array.BYTES_PER_ELEMENT)
    );

    const var_onnx__conv_1782 = builder.constant(
      { dataType: 'float16', shape: [192,1,1,32] },
      new Float16Array(weights_array_buffer, 78752, 12288 / Float16Array.BYTES_PER_ELEMENT)
    );

    const var_onnx__conv_1783 = builder.constant(
      { dataType: 'float16', shape: [192] },
      new Float16Array(weights_array_buffer, 91040, 384 / Float16Array.BYTES_PER_ELEMENT)
    );

    const var_onnx__conv_1785 = builder.constant(
      { dataType: 'float16', shape: [1,3,3,192] },
      new Float16Array(weights_array_buffer, 91440, 3456 / Float16Array.BYTES_PER_ELEMENT)
    );

    const var_onnx__conv_1786 = builder.constant(
      { dataType: 'float16', shape: [192] },
      new Float16Array(weights_array_buffer, 94896, 384 / Float16Array.BYTES_PER_ELEMENT)
    );

    const var_onnx__conv_1788 = builder.constant(
      { dataType: 'float16', shape: [32,1,1,192] },
      new Float16Array(weights_array_buffer, 95296, 12288 / Float16Array.BYTES_PER_ELEMENT)
    );

    const var_onnx__conv_1789 = builder.constant(
      { dataType: 'float16', shape: [32] },
      new Float16Array(weights_array_buffer, 107584, 64 / Float16Array.BYTES_PER_ELEMENT)
    );

    const var_onnx__conv_1791 = builder.constant(
      { dataType: 'float16', shape: [192,1,1,32] },
      new Float16Array(weights_array_buffer, 107648, 12288 / Float16Array.BYTES_PER_ELEMENT)
    );

    const var_onnx__conv_1792 = builder.constant(
      { dataType: 'float16', shape: [192] },
      new Float16Array(weights_array_buffer, 119936, 384 / Float16Array.BYTES_PER_ELEMENT)
    );

    const var_onnx__conv_1794 = builder.constant(
      { dataType: 'float16', shape: [1,3,3,192] },
      new Float16Array(weights_array_buffer, 120336, 3456 / Float16Array.BYTES_PER_ELEMENT)
    );

    const var_onnx__conv_1795 = builder.constant(
      { dataType: 'float16', shape: [192] },
      new Float16Array(weights_array_buffer, 123792, 384 / Float16Array.BYTES_PER_ELEMENT)
    );

    const var_onnx__conv_1797 = builder.constant(
      { dataType: 'float16', shape: [64,1,1,192] },
      new Float16Array(weights_array_buffer, 124192, 24576 / Float16Array.BYTES_PER_ELEMENT)
    );

    const var_onnx__conv_1798 = builder.constant(
      { dataType: 'float16', shape: [64] },
      new Float16Array(weights_array_buffer, 148768, 128 / Float16Array.BYTES_PER_ELEMENT)
    );

    const var_onnx__conv_1800 = builder.constant(
      { dataType: 'float16', shape: [384,1,1,64] },
      new Float16Array(weights_array_buffer, 148896, 49152 / Float16Array.BYTES_PER_ELEMENT)
    );

    const var_onnx__conv_1801 = builder.constant(
      { dataType: 'float16', shape: [384] },
      new Float16Array(weights_array_buffer, 198048, 768 / Float16Array.BYTES_PER_ELEMENT)
    );

    const var_onnx__conv_1803 = builder.constant(
      { dataType: 'float16', shape: [1,3,3,384] },
      new Float16Array(weights_array_buffer, 198832, 6912 / Float16Array.BYTES_PER_ELEMENT)
    );

    const var_onnx__conv_1804 = builder.constant(
      { dataType: 'float16', shape: [384] },
      new Float16Array(weights_array_buffer, 205744, 768 / Float16Array.BYTES_PER_ELEMENT)
    );

    const var_onnx__conv_1806 = builder.constant(
      { dataType: 'float16', shape: [64,1,1,384] },
      new Float16Array(weights_array_buffer, 206528, 49152 / Float16Array.BYTES_PER_ELEMENT)
    );

    const var_onnx__conv_1807 = builder.constant(
      { dataType: 'float16', shape: [64] },
      new Float16Array(weights_array_buffer, 255680, 128 / Float16Array.BYTES_PER_ELEMENT)
    );

    const var_onnx__conv_1809 = builder.constant(
      { dataType: 'float16', shape: [384,1,1,64] },
      new Float16Array(weights_array_buffer, 255808, 49152 / Float16Array.BYTES_PER_ELEMENT)
    );

    const var_onnx__conv_1810 = builder.constant(
      { dataType: 'float16', shape: [384] },
      new Float16Array(weights_array_buffer, 304960, 768 / Float16Array.BYTES_PER_ELEMENT)
    );

    const var_onnx__conv_1812 = builder.constant(
      { dataType: 'float16', shape: [1,3,3,384] },
      new Float16Array(weights_array_buffer, 305744, 6912 / Float16Array.BYTES_PER_ELEMENT)
    );

    const var_onnx__conv_1813 = builder.constant(
      { dataType: 'float16', shape: [384] },
      new Float16Array(weights_array_buffer, 312656, 768 / Float16Array.BYTES_PER_ELEMENT)
    );

    const var_onnx__conv_1815 = builder.constant(
      { dataType: 'float16', shape: [64,1,1,384] },
      new Float16Array(weights_array_buffer, 313440, 49152 / Float16Array.BYTES_PER_ELEMENT)
    );

    const var_onnx__conv_1816 = builder.constant(
      { dataType: 'float16', shape: [64] },
      new Float16Array(weights_array_buffer, 362592, 128 / Float16Array.BYTES_PER_ELEMENT)
    );

    const var_onnx__conv_1818 = builder.constant(
      { dataType: 'float16', shape: [384,1,1,64] },
      new Float16Array(weights_array_buffer, 362720, 49152 / Float16Array.BYTES_PER_ELEMENT)
    );

    const var_onnx__conv_1819 = builder.constant(
      { dataType: 'float16', shape: [384] },
      new Float16Array(weights_array_buffer, 411872, 768 / Float16Array.BYTES_PER_ELEMENT)
    );

    const var_onnx__conv_1821 = builder.constant(
      { dataType: 'float16', shape: [1,3,3,384] },
      new Float16Array(weights_array_buffer, 412656, 6912 / Float16Array.BYTES_PER_ELEMENT)
    );

    const var_onnx__conv_1822 = builder.constant(
      { dataType: 'float16', shape: [384] },
      new Float16Array(weights_array_buffer, 419568, 768 / Float16Array.BYTES_PER_ELEMENT)
    );

    const var_onnx__conv_1824 = builder.constant(
      { dataType: 'float16', shape: [64,1,1,384] },
      new Float16Array(weights_array_buffer, 420352, 49152 / Float16Array.BYTES_PER_ELEMENT)
    );

    const var_onnx__conv_1825 = builder.constant(
      { dataType: 'float16', shape: [64] },
      new Float16Array(weights_array_buffer, 469504, 128 / Float16Array.BYTES_PER_ELEMENT)
    );

    const var_onnx__conv_1827 = builder.constant(
      { dataType: 'float16', shape: [384,1,1,64] },
      new Float16Array(weights_array_buffer, 469632, 49152 / Float16Array.BYTES_PER_ELEMENT)
    );

    const var_onnx__conv_1828 = builder.constant(
      { dataType: 'float16', shape: [384] },
      new Float16Array(weights_array_buffer, 518784, 768 / Float16Array.BYTES_PER_ELEMENT)
    );

    const var_onnx__conv_1830 = builder.constant(
      { dataType: 'float16', shape: [1,3,3,384] },
      new Float16Array(weights_array_buffer, 519568, 6912 / Float16Array.BYTES_PER_ELEMENT)
    );

    const var_onnx__conv_1831 = builder.constant(
      { dataType: 'float16', shape: [384] },
      new Float16Array(weights_array_buffer, 526480, 768 / Float16Array.BYTES_PER_ELEMENT)
    );

    const var_onnx__conv_1833 = builder.constant(
      { dataType: 'float16', shape: [96,1,1,384] },
      new Float16Array(weights_array_buffer, 527264, 73728 / Float16Array.BYTES_PER_ELEMENT)
    );

    const var_onnx__conv_1834 = builder.constant(
      { dataType: 'float16', shape: [96] },
      new Float16Array(weights_array_buffer, 600992, 192 / Float16Array.BYTES_PER_ELEMENT)
    );

    const var_onnx__conv_1836 = builder.constant(
      { dataType: 'float16', shape: [576,1,1,96] },
      new Float16Array(weights_array_buffer, 601184, 110592 / Float16Array.BYTES_PER_ELEMENT)
    );

    const var_onnx__conv_1837 = builder.constant(
      { dataType: 'float16', shape: [576] },
      new Float16Array(weights_array_buffer, 711776, 1152 / Float16Array.BYTES_PER_ELEMENT)
    );

    const var_onnx__conv_1839 = builder.constant(
      { dataType: 'float16', shape: [1,3,3,576] },
      new Float16Array(weights_array_buffer, 712944, 10368 / Float16Array.BYTES_PER_ELEMENT)
    );

    const var_onnx__conv_1840 = builder.constant(
      { dataType: 'float16', shape: [576] },
      new Float16Array(weights_array_buffer, 723312, 1152 / Float16Array.BYTES_PER_ELEMENT)
    );

    const var_onnx__conv_1842 = builder.constant(
      { dataType: 'float16', shape: [96,1,1,576] },
      new Float16Array(weights_array_buffer, 724480, 110592 / Float16Array.BYTES_PER_ELEMENT)
    );

    const var_onnx__conv_1843 = builder.constant(
      { dataType: 'float16', shape: [96] },
      new Float16Array(weights_array_buffer, 835072, 192 / Float16Array.BYTES_PER_ELEMENT)
    );

    const var_onnx__conv_1845 = builder.constant(
      { dataType: 'float16', shape: [576,1,1,96] },
      new Float16Array(weights_array_buffer, 835264, 110592 / Float16Array.BYTES_PER_ELEMENT)
    );

    const var_onnx__conv_1846 = builder.constant(
      { dataType: 'float16', shape: [576] },
      new Float16Array(weights_array_buffer, 945856, 1152 / Float16Array.BYTES_PER_ELEMENT)
    );

    const var_onnx__conv_1848 = builder.constant(
      { dataType: 'float16', shape: [1,3,3,576] },
      new Float16Array(weights_array_buffer, 947024, 10368 / Float16Array.BYTES_PER_ELEMENT)
    );

    const var_onnx__conv_1849 = builder.constant(
      { dataType: 'float16', shape: [576] },
      new Float16Array(weights_array_buffer, 957392, 1152 / Float16Array.BYTES_PER_ELEMENT)
    );

    const var_onnx__conv_1851 = builder.constant(
      { dataType: 'float16', shape: [96,1,1,576] },
      new Float16Array(weights_array_buffer, 958560, 110592 / Float16Array.BYTES_PER_ELEMENT)
    );

    const var_onnx__conv_1852 = builder.constant(
      { dataType: 'float16', shape: [96] },
      new Float16Array(weights_array_buffer, 1069152, 192 / Float16Array.BYTES_PER_ELEMENT)
    );

    const var_onnx__conv_1854 = builder.constant(
      { dataType: 'float16', shape: [576,1,1,96] },
      new Float16Array(weights_array_buffer, 1069344, 110592 / Float16Array.BYTES_PER_ELEMENT)
    );

    const var_onnx__conv_1855 = builder.constant(
      { dataType: 'float16', shape: [576] },
      new Float16Array(weights_array_buffer, 1179936, 1152 / Float16Array.BYTES_PER_ELEMENT)
    );

    const var_onnx__conv_1857 = builder.constant(
      { dataType: 'float16', shape: [1,3,3,576] },
      new Float16Array(weights_array_buffer, 1181104, 10368 / Float16Array.BYTES_PER_ELEMENT)
    );

    const var_onnx__conv_1858 = builder.constant(
      { dataType: 'float16', shape: [576] },
      new Float16Array(weights_array_buffer, 1191472, 1152 / Float16Array.BYTES_PER_ELEMENT)
    );

    const var_onnx__conv_1860 = builder.constant(
      { dataType: 'float16', shape: [160,1,1,576] },
      new Float16Array(weights_array_buffer, 1192640, 184320 / Float16Array.BYTES_PER_ELEMENT)
    );

    const var_onnx__conv_1861 = builder.constant(
      { dataType: 'float16', shape: [160] },
      new Float16Array(weights_array_buffer, 1376960, 320 / Float16Array.BYTES_PER_ELEMENT)
    );

    const var_onnx__conv_1863 = builder.constant(
      { dataType: 'float16', shape: [960,1,1,160] },
      new Float16Array(weights_array_buffer, 1377280, 307200 / Float16Array.BYTES_PER_ELEMENT)
    );

    const var_onnx__conv_1864 = builder.constant(
      { dataType: 'float16', shape: [960] },
      new Float16Array(weights_array_buffer, 1684480, 1920 / Float16Array.BYTES_PER_ELEMENT)
    );

    const var_onnx__conv_1866 = builder.constant(
      { dataType: 'float16', shape: [1,3,3,960] },
      new Float16Array(weights_array_buffer, 1686416, 17280 / Float16Array.BYTES_PER_ELEMENT)
    );

    const var_onnx__conv_1867 = builder.constant(
      { dataType: 'float16', shape: [960] },
      new Float16Array(weights_array_buffer, 1703696, 1920 / Float16Array.BYTES_PER_ELEMENT)
    );

    const var_onnx__conv_1869 = builder.constant(
      { dataType: 'float16', shape: [160,1,1,960] },
      new Float16Array(weights_array_buffer, 1705632, 307200 / Float16Array.BYTES_PER_ELEMENT)
    );

    const var_onnx__conv_1870 = builder.constant(
      { dataType: 'float16', shape: [160] },
      new Float16Array(weights_array_buffer, 2012832, 320 / Float16Array.BYTES_PER_ELEMENT)
    );

    const var_onnx__conv_1872 = builder.constant(
      { dataType: 'float16', shape: [960,1,1,160] },
      new Float16Array(weights_array_buffer, 2013152, 307200 / Float16Array.BYTES_PER_ELEMENT)
    );

    const var_onnx__conv_1873 = builder.constant(
      { dataType: 'float16', shape: [960] },
      new Float16Array(weights_array_buffer, 2320352, 1920 / Float16Array.BYTES_PER_ELEMENT)
    );

    const var_onnx__conv_1875 = builder.constant(
      { dataType: 'float16', shape: [1,3,3,960] },
      new Float16Array(weights_array_buffer, 2322288, 17280 / Float16Array.BYTES_PER_ELEMENT)
    );

    const var_onnx__conv_1876 = builder.constant(
      { dataType: 'float16', shape: [960] },
      new Float16Array(weights_array_buffer, 2339568, 1920 / Float16Array.BYTES_PER_ELEMENT)
    );

    const var_onnx__conv_1878 = builder.constant(
      { dataType: 'float16', shape: [160,1,1,960] },
      new Float16Array(weights_array_buffer, 2341504, 307200 / Float16Array.BYTES_PER_ELEMENT)
    );

    const var_onnx__conv_1879 = builder.constant(
      { dataType: 'float16', shape: [160] },
      new Float16Array(weights_array_buffer, 2648704, 320 / Float16Array.BYTES_PER_ELEMENT)
    );

    const var_onnx__conv_1881 = builder.constant(
      { dataType: 'float16', shape: [960,1,1,160] },
      new Float16Array(weights_array_buffer, 2649024, 307200 / Float16Array.BYTES_PER_ELEMENT)
    );

    const var_onnx__conv_1882 = builder.constant(
      { dataType: 'float16', shape: [960] },
      new Float16Array(weights_array_buffer, 2956224, 1920 / Float16Array.BYTES_PER_ELEMENT)
    );

    const var_onnx__conv_1884 = builder.constant(
      { dataType: 'float16', shape: [1,3,3,960] },
      new Float16Array(weights_array_buffer, 2958160, 17280 / Float16Array.BYTES_PER_ELEMENT)
    );

    const var_onnx__conv_1885 = builder.constant(
      { dataType: 'float16', shape: [960] },
      new Float16Array(weights_array_buffer, 2975440, 1920 / Float16Array.BYTES_PER_ELEMENT)
    );

    const var_onnx__conv_1887 = builder.constant(
      { dataType: 'float16', shape: [320,1,1,960] },
      new Float16Array(weights_array_buffer, 2977376, 614400 / Float16Array.BYTES_PER_ELEMENT)
    );

    const var_onnx__conv_1888 = builder.constant(
      { dataType: 'float16', shape: [320] },
      new Float16Array(weights_array_buffer, 3591776, 640 / Float16Array.BYTES_PER_ELEMENT)
    );

    const var_onnx__conv_1890 = builder.constant(
      { dataType: 'float16', shape: [1280,1,1,320] },
      new Float16Array(weights_array_buffer, 3592416, 819200 / Float16Array.BYTES_PER_ELEMENT)
    );

    const var_onnx__conv_1891 = builder.constant(
      { dataType: 'float16', shape: [1280] },
      new Float16Array(weights_array_buffer, 4411616, 2560 / Float16Array.BYTES_PER_ELEMENT)
    );

    const var_classifier_weight = builder.constant(
      { dataType: 'float16', shape: [1001,1280] },
      new Float16Array(weights_array_buffer, 4414192, 2562560 / Float16Array.BYTES_PER_ELEMENT)
    );

    const var_classifier_bias = builder.constant(
      { dataType: 'float16', shape: [1001] },
      new Float16Array(weights_array_buffer, 6976752, 2002 / Float16Array.BYTES_PER_ELEMENT)
    );

    // Create graph operators
        
    const graph_input_cast_0 = builder.cast(
      pixel_values,
      'float16',
      { label: 'graph_input_cast0' }
    );
    
    const var__mobilenet_v2_conv_stem_first_conv_convolution_conv_output_0 = builder.conv2d(
      graph_input_cast_0, var_onnx__conv_1737,
      {
        strides: [2, 2],
        padding: [0, 1, 0, 1],
        dilations: [1, 1],
        groups: 1,
        bias: var_onnx__conv_1738,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: '/mobilenet_v2/conv_stem/first_conv/convolution/Conv'
      }
    );
    
    const var__mobilenet_v2_conv_stem_first_conv_activation_clip_output_0 = builder.clamp(
      var__mobilenet_v2_conv_stem_first_conv_convolution_conv_output_0,
      { label: '/mobilenet_v2/conv_stem/first_conv/activation/Clip' }
    );
    
    const var__mobilenet_v2_conv_stem_conv_3x3_convolution_conv_output_0 = builder.conv2d(
      var__mobilenet_v2_conv_stem_first_conv_activation_clip_output_0, var_onnx__conv_1740,
      {
        strides: [1, 1],
        padding: [1, 1, 1, 1],
        dilations: [1, 1],
        groups: 32,
        bias: var_onnx__conv_1741,
        filterLayout: 'ihwo',
        inputLayout: 'nhwc',
        label: '/mobilenet_v2/conv_stem/conv_3x3/convolution/Conv'
      }
    );
    
    const var__mobilenet_v2_conv_stem_conv_3x3_activation_clip_output_0 = builder.clamp(
      var__mobilenet_v2_conv_stem_conv_3x3_convolution_conv_output_0,
      { label: '/mobilenet_v2/conv_stem/conv_3x3/activation/Clip' }
    );
    
    const var__mobilenet_v2_conv_stem_reduce_1x1_convolution_conv_output_0 = builder.conv2d(
      var__mobilenet_v2_conv_stem_conv_3x3_activation_clip_output_0, var_onnx__conv_1743,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_onnx__conv_1744,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: '/mobilenet_v2/conv_stem/reduce_1x1/convolution/Conv'
      }
    );
    
    const var__mobilenet_v2_layer_0_expand_1x1_convolution_conv_output_0 = builder.conv2d(
      var__mobilenet_v2_conv_stem_reduce_1x1_convolution_conv_output_0, var_onnx__conv_1746,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_onnx__conv_1747,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: '/mobilenet_v2/layer.0/expand_1x1/convolution/Conv'
      }
    );
    
    const var__mobilenet_v2_layer_0_expand_1x1_activation_clip_output_0 = builder.clamp(
      var__mobilenet_v2_layer_0_expand_1x1_convolution_conv_output_0,
      { label: '/mobilenet_v2/layer.0/expand_1x1/activation/Clip' }
    );
    
    const var__mobilenet_v2_layer_0_conv_3x3_convolution_conv_output_0 = builder.conv2d(
      var__mobilenet_v2_layer_0_expand_1x1_activation_clip_output_0, var_onnx__conv_1749,
      {
        strides: [2, 2],
        padding: [0, 1, 0, 1],
        dilations: [1, 1],
        groups: 96,
        bias: var_onnx__conv_1750,
        filterLayout: 'ihwo',
        inputLayout: 'nhwc',
        label: '/mobilenet_v2/layer.0/conv_3x3/convolution/Conv'
      }
    );
    
    const var__mobilenet_v2_layer_0_conv_3x3_activation_clip_output_0 = builder.clamp(
      var__mobilenet_v2_layer_0_conv_3x3_convolution_conv_output_0,
      { label: '/mobilenet_v2/layer.0/conv_3x3/activation/Clip' }
    );
    
    const var__mobilenet_v2_layer_0_reduce_1x1_convolution_conv_output_0 = builder.conv2d(
      var__mobilenet_v2_layer_0_conv_3x3_activation_clip_output_0, var_onnx__conv_1752,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_onnx__conv_1753,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: '/mobilenet_v2/layer.0/reduce_1x1/convolution/Conv'
      }
    );
    
    const var__mobilenet_v2_layer_1_expand_1x1_convolution_conv_output_0 = builder.conv2d(
      var__mobilenet_v2_layer_0_reduce_1x1_convolution_conv_output_0, var_onnx__conv_1755,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_onnx__conv_1756,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: '/mobilenet_v2/layer.1/expand_1x1/convolution/Conv'
      }
    );
    
    const var__mobilenet_v2_layer_1_expand_1x1_activation_clip_output_0 = builder.clamp(
      var__mobilenet_v2_layer_1_expand_1x1_convolution_conv_output_0,
      { label: '/mobilenet_v2/layer.1/expand_1x1/activation/Clip' }
    );
    
    const var__mobilenet_v2_layer_1_conv_3x3_convolution_conv_output_0 = builder.conv2d(
      var__mobilenet_v2_layer_1_expand_1x1_activation_clip_output_0, var_onnx__conv_1758,
      {
        strides: [1, 1],
        padding: [1, 1, 1, 1],
        dilations: [1, 1],
        groups: 144,
        bias: var_onnx__conv_1759,
        filterLayout: 'ihwo',
        inputLayout: 'nhwc',
        label: '/mobilenet_v2/layer.1/conv_3x3/convolution/Conv'
      }
    );
    
    const var__mobilenet_v2_layer_1_conv_3x3_activation_clip_output_0 = builder.clamp(
      var__mobilenet_v2_layer_1_conv_3x3_convolution_conv_output_0,
      { label: '/mobilenet_v2/layer.1/conv_3x3/activation/Clip' }
    );
    
    const var__mobilenet_v2_layer_1_reduce_1x1_convolution_conv_output_0 = builder.conv2d(
      var__mobilenet_v2_layer_1_conv_3x3_activation_clip_output_0, var_onnx__conv_1761,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_onnx__conv_1762,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: '/mobilenet_v2/layer.1/reduce_1x1/convolution/Conv'
      }
    );
    
    const var__mobilenet_v2_layer_1_add_output_0 = builder.add(
      var__mobilenet_v2_layer_0_reduce_1x1_convolution_conv_output_0,
      var__mobilenet_v2_layer_1_reduce_1x1_convolution_conv_output_0,
      { label: '/mobilenet_v2/layer.1/Add' }
    );
    
    const var__mobilenet_v2_layer_2_expand_1x1_convolution_conv_output_0 = builder.conv2d(
      var__mobilenet_v2_layer_1_add_output_0, var_onnx__conv_1764,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_onnx__conv_1765,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: '/mobilenet_v2/layer.2/expand_1x1/convolution/Conv'
      }
    );
    
    const var__mobilenet_v2_layer_2_expand_1x1_activation_clip_output_0 = builder.clamp(
      var__mobilenet_v2_layer_2_expand_1x1_convolution_conv_output_0,
      { label: '/mobilenet_v2/layer.2/expand_1x1/activation/Clip' }
    );
    
    const var__mobilenet_v2_layer_2_conv_3x3_convolution_conv_output_0 = builder.conv2d(
      var__mobilenet_v2_layer_2_expand_1x1_activation_clip_output_0, var_onnx__conv_1767,
      {
        strides: [2, 2],
        padding: [0, 1, 0, 1],
        dilations: [1, 1],
        groups: 144,
        bias: var_onnx__conv_1768,
        filterLayout: 'ihwo',
        inputLayout: 'nhwc',
        label: '/mobilenet_v2/layer.2/conv_3x3/convolution/Conv'
      }
    );
    
    const var__mobilenet_v2_layer_2_conv_3x3_activation_clip_output_0 = builder.clamp(
      var__mobilenet_v2_layer_2_conv_3x3_convolution_conv_output_0,
      { label: '/mobilenet_v2/layer.2/conv_3x3/activation/Clip' }
    );
    
    const var__mobilenet_v2_layer_2_reduce_1x1_convolution_conv_output_0 = builder.conv2d(
      var__mobilenet_v2_layer_2_conv_3x3_activation_clip_output_0, var_onnx__conv_1770,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_onnx__conv_1771,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: '/mobilenet_v2/layer.2/reduce_1x1/convolution/Conv'
      }
    );
    
    const var__mobilenet_v2_layer_3_expand_1x1_convolution_conv_output_0 = builder.conv2d(
      var__mobilenet_v2_layer_2_reduce_1x1_convolution_conv_output_0, var_onnx__conv_1773,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_onnx__conv_1774,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: '/mobilenet_v2/layer.3/expand_1x1/convolution/Conv'
      }
    );
    
    const var__mobilenet_v2_layer_3_expand_1x1_activation_clip_output_0 = builder.clamp(
      var__mobilenet_v2_layer_3_expand_1x1_convolution_conv_output_0,
      { label: '/mobilenet_v2/layer.3/expand_1x1/activation/Clip' }
    );
    
    const var__mobilenet_v2_layer_3_conv_3x3_convolution_conv_output_0 = builder.conv2d(
      var__mobilenet_v2_layer_3_expand_1x1_activation_clip_output_0, var_onnx__conv_1776,
      {
        strides: [1, 1],
        padding: [1, 1, 1, 1],
        dilations: [1, 1],
        groups: 192,
        bias: var_onnx__conv_1777,
        filterLayout: 'ihwo',
        inputLayout: 'nhwc',
        label: '/mobilenet_v2/layer.3/conv_3x3/convolution/Conv'
      }
    );
    
    const var__mobilenet_v2_layer_3_conv_3x3_activation_clip_output_0 = builder.clamp(
      var__mobilenet_v2_layer_3_conv_3x3_convolution_conv_output_0,
      { label: '/mobilenet_v2/layer.3/conv_3x3/activation/Clip' }
    );
    
    const var__mobilenet_v2_layer_3_reduce_1x1_convolution_conv_output_0 = builder.conv2d(
      var__mobilenet_v2_layer_3_conv_3x3_activation_clip_output_0, var_onnx__conv_1779,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_onnx__conv_1780,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: '/mobilenet_v2/layer.3/reduce_1x1/convolution/Conv'
      }
    );
    
    const var__mobilenet_v2_layer_3_add_output_0 = builder.add(
      var__mobilenet_v2_layer_2_reduce_1x1_convolution_conv_output_0,
      var__mobilenet_v2_layer_3_reduce_1x1_convolution_conv_output_0,
      { label: '/mobilenet_v2/layer.3/Add' }
    );
    
    const var__mobilenet_v2_layer_4_expand_1x1_convolution_conv_output_0 = builder.conv2d(
      var__mobilenet_v2_layer_3_add_output_0, var_onnx__conv_1782,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_onnx__conv_1783,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: '/mobilenet_v2/layer.4/expand_1x1/convolution/Conv'
      }
    );
    
    const var__mobilenet_v2_layer_4_expand_1x1_activation_clip_output_0 = builder.clamp(
      var__mobilenet_v2_layer_4_expand_1x1_convolution_conv_output_0,
      { label: '/mobilenet_v2/layer.4/expand_1x1/activation/Clip' }
    );
    
    const var__mobilenet_v2_layer_4_conv_3x3_convolution_conv_output_0 = builder.conv2d(
      var__mobilenet_v2_layer_4_expand_1x1_activation_clip_output_0, var_onnx__conv_1785,
      {
        strides: [1, 1],
        padding: [1, 1, 1, 1],
        dilations: [1, 1],
        groups: 192,
        bias: var_onnx__conv_1786,
        filterLayout: 'ihwo',
        inputLayout: 'nhwc',
        label: '/mobilenet_v2/layer.4/conv_3x3/convolution/Conv'
      }
    );
    
    const var__mobilenet_v2_layer_4_conv_3x3_activation_clip_output_0 = builder.clamp(
      var__mobilenet_v2_layer_4_conv_3x3_convolution_conv_output_0,
      { label: '/mobilenet_v2/layer.4/conv_3x3/activation/Clip' }
    );
    
    const var__mobilenet_v2_layer_4_reduce_1x1_convolution_conv_output_0 = builder.conv2d(
      var__mobilenet_v2_layer_4_conv_3x3_activation_clip_output_0, var_onnx__conv_1788,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_onnx__conv_1789,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: '/mobilenet_v2/layer.4/reduce_1x1/convolution/Conv'
      }
    );
    
    const var__mobilenet_v2_layer_4_add_output_0 = builder.add(
      var__mobilenet_v2_layer_3_add_output_0,
      var__mobilenet_v2_layer_4_reduce_1x1_convolution_conv_output_0,
      { label: '/mobilenet_v2/layer.4/Add' }
    );
    
    const var__mobilenet_v2_layer_5_expand_1x1_convolution_conv_output_0 = builder.conv2d(
      var__mobilenet_v2_layer_4_add_output_0, var_onnx__conv_1791,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_onnx__conv_1792,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: '/mobilenet_v2/layer.5/expand_1x1/convolution/Conv'
      }
    );
    
    const var__mobilenet_v2_layer_5_expand_1x1_activation_clip_output_0 = builder.clamp(
      var__mobilenet_v2_layer_5_expand_1x1_convolution_conv_output_0,
      { label: '/mobilenet_v2/layer.5/expand_1x1/activation/Clip' }
    );
    
    const var__mobilenet_v2_layer_5_conv_3x3_convolution_conv_output_0 = builder.conv2d(
      var__mobilenet_v2_layer_5_expand_1x1_activation_clip_output_0, var_onnx__conv_1794,
      {
        strides: [2, 2],
        padding: [0, 1, 0, 1],
        dilations: [1, 1],
        groups: 192,
        bias: var_onnx__conv_1795,
        filterLayout: 'ihwo',
        inputLayout: 'nhwc',
        label: '/mobilenet_v2/layer.5/conv_3x3/convolution/Conv'
      }
    );
    
    const var__mobilenet_v2_layer_5_conv_3x3_activation_clip_output_0 = builder.clamp(
      var__mobilenet_v2_layer_5_conv_3x3_convolution_conv_output_0,
      { label: '/mobilenet_v2/layer.5/conv_3x3/activation/Clip' }
    );
    
    const var__mobilenet_v2_layer_5_reduce_1x1_convolution_conv_output_0 = builder.conv2d(
      var__mobilenet_v2_layer_5_conv_3x3_activation_clip_output_0, var_onnx__conv_1797,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_onnx__conv_1798,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: '/mobilenet_v2/layer.5/reduce_1x1/convolution/Conv'
      }
    );
    
    const var__mobilenet_v2_layer_6_expand_1x1_convolution_conv_output_0 = builder.conv2d(
      var__mobilenet_v2_layer_5_reduce_1x1_convolution_conv_output_0, var_onnx__conv_1800,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_onnx__conv_1801,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: '/mobilenet_v2/layer.6/expand_1x1/convolution/Conv'
      }
    );
    
    const var__mobilenet_v2_layer_6_expand_1x1_activation_clip_output_0 = builder.clamp(
      var__mobilenet_v2_layer_6_expand_1x1_convolution_conv_output_0,
      { label: '/mobilenet_v2/layer.6/expand_1x1/activation/Clip' }
    );
    
    const var__mobilenet_v2_layer_6_conv_3x3_convolution_conv_output_0 = builder.conv2d(
      var__mobilenet_v2_layer_6_expand_1x1_activation_clip_output_0, var_onnx__conv_1803,
      {
        strides: [1, 1],
        padding: [1, 1, 1, 1],
        dilations: [1, 1],
        groups: 384,
        bias: var_onnx__conv_1804,
        filterLayout: 'ihwo',
        inputLayout: 'nhwc',
        label: '/mobilenet_v2/layer.6/conv_3x3/convolution/Conv'
      }
    );
    
    const var__mobilenet_v2_layer_6_conv_3x3_activation_clip_output_0 = builder.clamp(
      var__mobilenet_v2_layer_6_conv_3x3_convolution_conv_output_0,
      { label: '/mobilenet_v2/layer.6/conv_3x3/activation/Clip' }
    );
    
    const var__mobilenet_v2_layer_6_reduce_1x1_convolution_conv_output_0 = builder.conv2d(
      var__mobilenet_v2_layer_6_conv_3x3_activation_clip_output_0, var_onnx__conv_1806,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_onnx__conv_1807,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: '/mobilenet_v2/layer.6/reduce_1x1/convolution/Conv'
      }
    );
    
    const var__mobilenet_v2_layer_6_add_output_0 = builder.add(
      var__mobilenet_v2_layer_5_reduce_1x1_convolution_conv_output_0,
      var__mobilenet_v2_layer_6_reduce_1x1_convolution_conv_output_0,
      { label: '/mobilenet_v2/layer.6/Add' }
    );
    
    const var__mobilenet_v2_layer_7_expand_1x1_convolution_conv_output_0 = builder.conv2d(
      var__mobilenet_v2_layer_6_add_output_0, var_onnx__conv_1809,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_onnx__conv_1810,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: '/mobilenet_v2/layer.7/expand_1x1/convolution/Conv'
      }
    );
    
    const var__mobilenet_v2_layer_7_expand_1x1_activation_clip_output_0 = builder.clamp(
      var__mobilenet_v2_layer_7_expand_1x1_convolution_conv_output_0,
      { label: '/mobilenet_v2/layer.7/expand_1x1/activation/Clip' }
    );
    
    const var__mobilenet_v2_layer_7_conv_3x3_convolution_conv_output_0 = builder.conv2d(
      var__mobilenet_v2_layer_7_expand_1x1_activation_clip_output_0, var_onnx__conv_1812,
      {
        strides: [1, 1],
        padding: [1, 1, 1, 1],
        dilations: [1, 1],
        groups: 384,
        bias: var_onnx__conv_1813,
        filterLayout: 'ihwo',
        inputLayout: 'nhwc',
        label: '/mobilenet_v2/layer.7/conv_3x3/convolution/Conv'
      }
    );
    
    const var__mobilenet_v2_layer_7_conv_3x3_activation_clip_output_0 = builder.clamp(
      var__mobilenet_v2_layer_7_conv_3x3_convolution_conv_output_0,
      { label: '/mobilenet_v2/layer.7/conv_3x3/activation/Clip' }
    );
    
    const var__mobilenet_v2_layer_7_reduce_1x1_convolution_conv_output_0 = builder.conv2d(
      var__mobilenet_v2_layer_7_conv_3x3_activation_clip_output_0, var_onnx__conv_1815,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_onnx__conv_1816,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: '/mobilenet_v2/layer.7/reduce_1x1/convolution/Conv'
      }
    );
    
    const var__mobilenet_v2_layer_7_add_output_0 = builder.add(
      var__mobilenet_v2_layer_6_add_output_0,
      var__mobilenet_v2_layer_7_reduce_1x1_convolution_conv_output_0,
      { label: '/mobilenet_v2/layer.7/Add' }
    );
    
    const var__mobilenet_v2_layer_8_expand_1x1_convolution_conv_output_0 = builder.conv2d(
      var__mobilenet_v2_layer_7_add_output_0, var_onnx__conv_1818,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_onnx__conv_1819,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: '/mobilenet_v2/layer.8/expand_1x1/convolution/Conv'
      }
    );
    
    const var__mobilenet_v2_layer_8_expand_1x1_activation_clip_output_0 = builder.clamp(
      var__mobilenet_v2_layer_8_expand_1x1_convolution_conv_output_0,
      { label: '/mobilenet_v2/layer.8/expand_1x1/activation/Clip' }
    );
    
    const var__mobilenet_v2_layer_8_conv_3x3_convolution_conv_output_0 = builder.conv2d(
      var__mobilenet_v2_layer_8_expand_1x1_activation_clip_output_0, var_onnx__conv_1821,
      {
        strides: [1, 1],
        padding: [1, 1, 1, 1],
        dilations: [1, 1],
        groups: 384,
        bias: var_onnx__conv_1822,
        filterLayout: 'ihwo',
        inputLayout: 'nhwc',
        label: '/mobilenet_v2/layer.8/conv_3x3/convolution/Conv'
      }
    );
    
    const var__mobilenet_v2_layer_8_conv_3x3_activation_clip_output_0 = builder.clamp(
      var__mobilenet_v2_layer_8_conv_3x3_convolution_conv_output_0,
      { label: '/mobilenet_v2/layer.8/conv_3x3/activation/Clip' }
    );
    
    const var__mobilenet_v2_layer_8_reduce_1x1_convolution_conv_output_0 = builder.conv2d(
      var__mobilenet_v2_layer_8_conv_3x3_activation_clip_output_0, var_onnx__conv_1824,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_onnx__conv_1825,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: '/mobilenet_v2/layer.8/reduce_1x1/convolution/Conv'
      }
    );
    
    const var__mobilenet_v2_layer_8_add_output_0 = builder.add(
      var__mobilenet_v2_layer_7_add_output_0,
      var__mobilenet_v2_layer_8_reduce_1x1_convolution_conv_output_0,
      { label: '/mobilenet_v2/layer.8/Add' }
    );
    
    const var__mobilenet_v2_layer_9_expand_1x1_convolution_conv_output_0 = builder.conv2d(
      var__mobilenet_v2_layer_8_add_output_0, var_onnx__conv_1827,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_onnx__conv_1828,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: '/mobilenet_v2/layer.9/expand_1x1/convolution/Conv'
      }
    );
    
    const var__mobilenet_v2_layer_9_expand_1x1_activation_clip_output_0 = builder.clamp(
      var__mobilenet_v2_layer_9_expand_1x1_convolution_conv_output_0,
      { label: '/mobilenet_v2/layer.9/expand_1x1/activation/Clip' }
    );
    
    const var__mobilenet_v2_layer_9_conv_3x3_convolution_conv_output_0 = builder.conv2d(
      var__mobilenet_v2_layer_9_expand_1x1_activation_clip_output_0, var_onnx__conv_1830,
      {
        strides: [1, 1],
        padding: [1, 1, 1, 1],
        dilations: [1, 1],
        groups: 384,
        bias: var_onnx__conv_1831,
        filterLayout: 'ihwo',
        inputLayout: 'nhwc',
        label: '/mobilenet_v2/layer.9/conv_3x3/convolution/Conv'
      }
    );
    
    const var__mobilenet_v2_layer_9_conv_3x3_activation_clip_output_0 = builder.clamp(
      var__mobilenet_v2_layer_9_conv_3x3_convolution_conv_output_0,
      { label: '/mobilenet_v2/layer.9/conv_3x3/activation/Clip' }
    );
    
    const var__mobilenet_v2_layer_9_reduce_1x1_convolution_conv_output_0 = builder.conv2d(
      var__mobilenet_v2_layer_9_conv_3x3_activation_clip_output_0, var_onnx__conv_1833,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_onnx__conv_1834,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: '/mobilenet_v2/layer.9/reduce_1x1/convolution/Conv'
      }
    );
    
    const var__mobilenet_v2_layer_10_expand_1x1_convolution_conv_output_0 = builder.conv2d(
      var__mobilenet_v2_layer_9_reduce_1x1_convolution_conv_output_0, var_onnx__conv_1836,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_onnx__conv_1837,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: '/mobilenet_v2/layer.10/expand_1x1/convolution/Conv'
      }
    );
    
    const var__mobilenet_v2_layer_10_expand_1x1_activation_clip_output_0 = builder.clamp(
      var__mobilenet_v2_layer_10_expand_1x1_convolution_conv_output_0,
      { label: '/mobilenet_v2/layer.10/expand_1x1/activation/Clip' }
    );
    
    const var__mobilenet_v2_layer_10_conv_3x3_convolution_conv_output_0 = builder.conv2d(
      var__mobilenet_v2_layer_10_expand_1x1_activation_clip_output_0, var_onnx__conv_1839,
      {
        strides: [1, 1],
        padding: [1, 1, 1, 1],
        dilations: [1, 1],
        groups: 576,
        bias: var_onnx__conv_1840,
        filterLayout: 'ihwo',
        inputLayout: 'nhwc',
        label: '/mobilenet_v2/layer.10/conv_3x3/convolution/Conv'
      }
    );
    
    const var__mobilenet_v2_layer_10_conv_3x3_activation_clip_output_0 = builder.clamp(
      var__mobilenet_v2_layer_10_conv_3x3_convolution_conv_output_0,
      { label: '/mobilenet_v2/layer.10/conv_3x3/activation/Clip' }
    );
    
    const var__mobilenet_v2_layer_10_reduce_1x1_convolution_conv_output_0 = builder.conv2d(
      var__mobilenet_v2_layer_10_conv_3x3_activation_clip_output_0, var_onnx__conv_1842,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_onnx__conv_1843,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: '/mobilenet_v2/layer.10/reduce_1x1/convolution/Conv'
      }
    );
    
    const var__mobilenet_v2_layer_10_add_output_0 = builder.add(
      var__mobilenet_v2_layer_9_reduce_1x1_convolution_conv_output_0,
      var__mobilenet_v2_layer_10_reduce_1x1_convolution_conv_output_0,
      { label: '/mobilenet_v2/layer.10/Add' }
    );
    
    const var__mobilenet_v2_layer_11_expand_1x1_convolution_conv_output_0 = builder.conv2d(
      var__mobilenet_v2_layer_10_add_output_0, var_onnx__conv_1845,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_onnx__conv_1846,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: '/mobilenet_v2/layer.11/expand_1x1/convolution/Conv'
      }
    );
    
    const var__mobilenet_v2_layer_11_expand_1x1_activation_clip_output_0 = builder.clamp(
      var__mobilenet_v2_layer_11_expand_1x1_convolution_conv_output_0,
      { label: '/mobilenet_v2/layer.11/expand_1x1/activation/Clip' }
    );
    
    const var__mobilenet_v2_layer_11_conv_3x3_convolution_conv_output_0 = builder.conv2d(
      var__mobilenet_v2_layer_11_expand_1x1_activation_clip_output_0, var_onnx__conv_1848,
      {
        strides: [1, 1],
        padding: [1, 1, 1, 1],
        dilations: [1, 1],
        groups: 576,
        bias: var_onnx__conv_1849,
        filterLayout: 'ihwo',
        inputLayout: 'nhwc',
        label: '/mobilenet_v2/layer.11/conv_3x3/convolution/Conv'
      }
    );
    
    const var__mobilenet_v2_layer_11_conv_3x3_activation_clip_output_0 = builder.clamp(
      var__mobilenet_v2_layer_11_conv_3x3_convolution_conv_output_0,
      { label: '/mobilenet_v2/layer.11/conv_3x3/activation/Clip' }
    );
    
    const var__mobilenet_v2_layer_11_reduce_1x1_convolution_conv_output_0 = builder.conv2d(
      var__mobilenet_v2_layer_11_conv_3x3_activation_clip_output_0, var_onnx__conv_1851,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_onnx__conv_1852,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: '/mobilenet_v2/layer.11/reduce_1x1/convolution/Conv'
      }
    );
    
    const var__mobilenet_v2_layer_11_add_output_0 = builder.add(
      var__mobilenet_v2_layer_10_add_output_0,
      var__mobilenet_v2_layer_11_reduce_1x1_convolution_conv_output_0,
      { label: '/mobilenet_v2/layer.11/Add' }
    );
    
    const var__mobilenet_v2_layer_12_expand_1x1_convolution_conv_output_0 = builder.conv2d(
      var__mobilenet_v2_layer_11_add_output_0, var_onnx__conv_1854,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_onnx__conv_1855,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: '/mobilenet_v2/layer.12/expand_1x1/convolution/Conv'
      }
    );
    
    const var__mobilenet_v2_layer_12_expand_1x1_activation_clip_output_0 = builder.clamp(
      var__mobilenet_v2_layer_12_expand_1x1_convolution_conv_output_0,
      { label: '/mobilenet_v2/layer.12/expand_1x1/activation/Clip' }
    );
    
    const var__mobilenet_v2_layer_12_conv_3x3_convolution_conv_output_0 = builder.conv2d(
      var__mobilenet_v2_layer_12_expand_1x1_activation_clip_output_0, var_onnx__conv_1857,
      {
        strides: [2, 2],
        padding: [0, 1, 0, 1],
        dilations: [1, 1],
        groups: 576,
        bias: var_onnx__conv_1858,
        filterLayout: 'ihwo',
        inputLayout: 'nhwc',
        label: '/mobilenet_v2/layer.12/conv_3x3/convolution/Conv'
      }
    );
    
    const var__mobilenet_v2_layer_12_conv_3x3_activation_clip_output_0 = builder.clamp(
      var__mobilenet_v2_layer_12_conv_3x3_convolution_conv_output_0,
      { label: '/mobilenet_v2/layer.12/conv_3x3/activation/Clip' }
    );
    
    const var__mobilenet_v2_layer_12_reduce_1x1_convolution_conv_output_0 = builder.conv2d(
      var__mobilenet_v2_layer_12_conv_3x3_activation_clip_output_0, var_onnx__conv_1860,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_onnx__conv_1861,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: '/mobilenet_v2/layer.12/reduce_1x1/convolution/Conv'
      }
    );
    
    const var__mobilenet_v2_layer_13_expand_1x1_convolution_conv_output_0 = builder.conv2d(
      var__mobilenet_v2_layer_12_reduce_1x1_convolution_conv_output_0, var_onnx__conv_1863,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_onnx__conv_1864,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: '/mobilenet_v2/layer.13/expand_1x1/convolution/Conv'
      }
    );
    
    const var__mobilenet_v2_layer_13_expand_1x1_activation_clip_output_0 = builder.clamp(
      var__mobilenet_v2_layer_13_expand_1x1_convolution_conv_output_0,
      { label: '/mobilenet_v2/layer.13/expand_1x1/activation/Clip' }
    );
    
    const var__mobilenet_v2_layer_13_conv_3x3_convolution_conv_output_0 = builder.conv2d(
      var__mobilenet_v2_layer_13_expand_1x1_activation_clip_output_0, var_onnx__conv_1866,
      {
        strides: [1, 1],
        padding: [1, 1, 1, 1],
        dilations: [1, 1],
        groups: 960,
        bias: var_onnx__conv_1867,
        filterLayout: 'ihwo',
        inputLayout: 'nhwc',
        label: '/mobilenet_v2/layer.13/conv_3x3/convolution/Conv'
      }
    );
    
    const var__mobilenet_v2_layer_13_conv_3x3_activation_clip_output_0 = builder.clamp(
      var__mobilenet_v2_layer_13_conv_3x3_convolution_conv_output_0,
      { label: '/mobilenet_v2/layer.13/conv_3x3/activation/Clip' }
    );
    
    const var__mobilenet_v2_layer_13_reduce_1x1_convolution_conv_output_0 = builder.conv2d(
      var__mobilenet_v2_layer_13_conv_3x3_activation_clip_output_0, var_onnx__conv_1869,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_onnx__conv_1870,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: '/mobilenet_v2/layer.13/reduce_1x1/convolution/Conv'
      }
    );
    
    const var__mobilenet_v2_layer_13_add_output_0 = builder.add(
      var__mobilenet_v2_layer_12_reduce_1x1_convolution_conv_output_0,
      var__mobilenet_v2_layer_13_reduce_1x1_convolution_conv_output_0,
      { label: '/mobilenet_v2/layer.13/Add' }
    );
    
    const var__mobilenet_v2_layer_14_expand_1x1_convolution_conv_output_0 = builder.conv2d(
      var__mobilenet_v2_layer_13_add_output_0, var_onnx__conv_1872,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_onnx__conv_1873,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: '/mobilenet_v2/layer.14/expand_1x1/convolution/Conv'
      }
    );
    
    const var__mobilenet_v2_layer_14_expand_1x1_activation_clip_output_0 = builder.clamp(
      var__mobilenet_v2_layer_14_expand_1x1_convolution_conv_output_0,
      { label: '/mobilenet_v2/layer.14/expand_1x1/activation/Clip' }
    );
    
    const var__mobilenet_v2_layer_14_conv_3x3_convolution_conv_output_0 = builder.conv2d(
      var__mobilenet_v2_layer_14_expand_1x1_activation_clip_output_0, var_onnx__conv_1875,
      {
        strides: [1, 1],
        padding: [1, 1, 1, 1],
        dilations: [1, 1],
        groups: 960,
        bias: var_onnx__conv_1876,
        filterLayout: 'ihwo',
        inputLayout: 'nhwc',
        label: '/mobilenet_v2/layer.14/conv_3x3/convolution/Conv'
      }
    );
    
    const var__mobilenet_v2_layer_14_conv_3x3_activation_clip_output_0 = builder.clamp(
      var__mobilenet_v2_layer_14_conv_3x3_convolution_conv_output_0,
      { label: '/mobilenet_v2/layer.14/conv_3x3/activation/Clip' }
    );
    
    const var__mobilenet_v2_layer_14_reduce_1x1_convolution_conv_output_0 = builder.conv2d(
      var__mobilenet_v2_layer_14_conv_3x3_activation_clip_output_0, var_onnx__conv_1878,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_onnx__conv_1879,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: '/mobilenet_v2/layer.14/reduce_1x1/convolution/Conv'
      }
    );
    
    const var__mobilenet_v2_layer_14_add_output_0 = builder.add(
      var__mobilenet_v2_layer_13_add_output_0,
      var__mobilenet_v2_layer_14_reduce_1x1_convolution_conv_output_0,
      { label: '/mobilenet_v2/layer.14/Add' }
    );
    
    const var__mobilenet_v2_layer_15_expand_1x1_convolution_conv_output_0 = builder.conv2d(
      var__mobilenet_v2_layer_14_add_output_0, var_onnx__conv_1881,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_onnx__conv_1882,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: '/mobilenet_v2/layer.15/expand_1x1/convolution/Conv'
      }
    );
    
    const var__mobilenet_v2_layer_15_expand_1x1_activation_clip_output_0 = builder.clamp(
      var__mobilenet_v2_layer_15_expand_1x1_convolution_conv_output_0,
      { label: '/mobilenet_v2/layer.15/expand_1x1/activation/Clip' }
    );
    
    const var__mobilenet_v2_layer_15_conv_3x3_convolution_conv_output_0 = builder.conv2d(
      var__mobilenet_v2_layer_15_expand_1x1_activation_clip_output_0, var_onnx__conv_1884,
      {
        strides: [1, 1],
        padding: [1, 1, 1, 1],
        dilations: [1, 1],
        groups: 960,
        bias: var_onnx__conv_1885,
        filterLayout: 'ihwo',
        inputLayout: 'nhwc',
        label: '/mobilenet_v2/layer.15/conv_3x3/convolution/Conv'
      }
    );
    
    const var__mobilenet_v2_layer_15_conv_3x3_activation_clip_output_0 = builder.clamp(
      var__mobilenet_v2_layer_15_conv_3x3_convolution_conv_output_0,
      { label: '/mobilenet_v2/layer.15/conv_3x3/activation/Clip' }
    );
    
    const var__mobilenet_v2_layer_15_reduce_1x1_convolution_conv_output_0 = builder.conv2d(
      var__mobilenet_v2_layer_15_conv_3x3_activation_clip_output_0, var_onnx__conv_1887,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_onnx__conv_1888,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: '/mobilenet_v2/layer.15/reduce_1x1/convolution/Conv'
      }
    );
    
    const var__mobilenet_v2_conv_1x1_convolution_conv_output_0 = builder.conv2d(
      var__mobilenet_v2_layer_15_reduce_1x1_convolution_conv_output_0, var_onnx__conv_1890,
      {
        strides: [1, 1],
        padding: [0, 0, 0, 0],
        dilations: [1, 1],
        groups: 1,
        bias: var_onnx__conv_1891,
        filterLayout: 'ohwi',
        inputLayout: 'nhwc',
        label: '/mobilenet_v2/conv_1x1/convolution/Conv'
      }
    );
    
    const var__mobilenet_v2_conv_1x1_activation_clip_output_0 = builder.clamp(
      var__mobilenet_v2_conv_1x1_convolution_conv_output_0,
      { label: '/mobilenet_v2/conv_1x1/activation/Clip' }
    );
    
    const var__mobilenet_v2_pooler_globalaveragepool_output_0 = builder.averagePool2d(
      var__mobilenet_v2_conv_1x1_activation_clip_output_0,
      {
        layout: 'nhwc',
        label: '/mobilenet_v2/pooler/GlobalAveragePool'
      }
    );
    
    const var__mobilenet_v2_flatten_output_0 = builder.reshape(
      var__mobilenet_v2_pooler_globalaveragepool_output_0,
      [1, 1280], 
      { label: '/mobilenet_v2/Flatten' }
    );
    
    const graph_output_cast_0 = builder.gemm(
      var__mobilenet_v2_flatten_output_0,
      var_classifier_weight,
      {
        alpha: 1.0,
        beta: 1.0,
        aTranspose: false,
        bTranspose: true,
        C: var_classifier_bias
      }
    );
    
    const logits = builder.cast(
      graph_output_cast_0,
      'float32',
      { label: 'graph_output_cast0' }
    );

    // Build graph with all outputs
    
    this.graph_ = await builder.build({ 'logits': logits });

    // Create output tensors
    
    this.outputTensors_['logits'] = await this.context_.createTensor(
      { dataType: 'float32', shape: [1,1001], readable: true }
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