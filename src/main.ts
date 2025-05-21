/**
 * Main entry point for WebNN Code Generator
 */
import './style.css';
import webnnLogo from '/logo/webnn.svg';
import githubLogo from '/logo/github.svg?raw';
import { initializeCodeGenerator } from './ui';
import { initializeInterface } from './ui';

/**
 * Initialize the application
 */
const initializeApp = (): void => {
  renderAppLayout();
  initializeInterface();
  initializeCodeGenerator(document.querySelector<HTMLButtonElement>('#generate-btn'));
};

/**
 * Render the main application template
 */
const renderAppLayout = (): void => {
  const appContainer = document.querySelector<HTMLDivElement>('#app');
  if (!appContainer) return;

  appContainer.innerHTML = `
    <div class="app-container">
      <header>
        <a href="https://github.com/ibelem/webnn-code-generator" class="logo-link">
          <img src="${webnnLogo}" class="logo" alt="WebNN logo" />
        </a>
        <h1>WebNN Code Generator</h1>
      </header>
      
      <div class="file-upload-panel">
        <div class="step step-1">
          <div class="step-icon">1</div>
          <div id="step-1" title="Convert your ONNX, TensorFlow Lite, or other model formats into Graph, Weight and BIN files">
            Donwload Graph, Weight and BIN files by using <a href="https://ibelem.github.io/netron">WebNN Netron</a>
          </div>
        </div>
        <div class="step step-2">
          <div class="step-icon">2</div>
          <div id="step-2">
            <div class="upload-item">
              <label for="graph-file-input" class="upload-button">Choose Graph File</label>
              <input type="file" id="graph-file-input" accept=".json">
              <span class="file-info" id="graph-file-info">No .json file selected</span>
            </div>
            
            <div class="upload-item">
              <label for="weight-file-input" class="upload-button">Choose Weights File</label>
              <input type="file" id="weight-file-input" accept=".json">
              <span class="file-info" id="weight-file-info">No .json file selected</span>
            </div>
            
            <div class="upload-item">
              <label for="bin-file-input" class="upload-button">Choose Bin File</label>
              <input type="file" id="bin-file-input" accept=".bin">
              <span class="file-info" id="bin-file-info">No .bin file selected</span>
            </div>
          </div>
        </div>
        
        <div class="step step-3">
          <div class="step-icon">3</div>
          <div id="step-3" class="upload-item generate-action">
            <button id="generate-btn" type="button" disabled>Generate WebNN Code</button>
          </div>
        </div>
      </div>
      <div id="free-dimension-overrides" class="override"></div>
      <div class="output-panel">
        <div id="output-graph" class="panel"></div>
        <div id="output-weight" class="panel"></div>
        <div id="output-code" class="code panel"></div>
      </div>
      <div id="log-console" class="status"></div>
      <div class="app-description">
        <ul>
          <li>Generate WebNN API code in vanilla JavaScript from ONNX, TensorFlow Lite, or other model formats.</li>
          <li>All model conversion and code generation processes execute entirely within your browser, ensuring your intellectual property remains private and secure as no model data is transmitted to or stored on cloud servers.</li>
        </ul>
      </div>
      <footer>
        &copy;2025 <a href="https://ibelem.github.io/webnn-code-generator/" title="WebNN Code Generator">WebNN Code Generator</a> · <a href="https://github.com/ibelem/webnn-code-generator/issues" title="WebNN Code Generator Issues">${githubLogo}</a> · <a href="http://ibelem.github.io/webnn-code-generator/?graph=https://ibelem.github.io/webnn-code-generator/model/mobilenetv2-12-static/graph.json&weights=https://ibelem.github.io/webnn-code-generator/model/mobilenetv2-12-static/weights.json&bin=https://ibelem.github.io/webnn-code-generator/model/mobilenetv2-12-static/weights.bin">Example</a> · <a href="https://ibelem.github.io/netron/" title="WebNN Netron">WebNN Netron</a> · <a href="https://github.com/huningxin/onnx2json/tree/webnn_js" title="Create WebNN JavaScript code for ONNX model">ONNX to JSON to WebNN</a>
      </footer>
    </div>
  `;
};

// Initialize the application when the DOM is ready
document.addEventListener('DOMContentLoaded', initializeApp);