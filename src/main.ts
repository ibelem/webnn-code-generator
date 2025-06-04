/**
 * Main entry point for WebNN Code Generator
 */
import './style.css';
import webnnLogo from '/logo/webnn.svg';
import githubLogo from '/logo/github.svg?raw';
import downloadLogo from '/logo/download.svg?raw';
import codeLogo from '/logo/code.svg?raw';
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
      <header title="Generate WebNN API code in vanilla JavaScript from ONNX, TensorFlow Lite, or other model formats.">
        <a href="https://github.com/ibelem/webnn-code-generator" class="logo-link">
          <img src="${webnnLogo}" class="logo" alt="WebNN logo" />
        </a>
        <h1>WebNN Code Generator</h1>
      </header>
      
      <div class="file-upload-panel">
        <div class="step step-1 disabled">
          <div class="step-icon">1</div>
          <div id="step-1" title="Convert your ONNX, TensorFlow Lite, or other model formats into graph, weights and bin files">
            Donwload graph and weights files by using <a href="https://ibelem.github.io/netron">WebNN Netron</a>
          </div>
        </div>
        <div class="step step-2 disabled">
          <div class="step-icon">2</div>
          <div id="step-2">
            <div class="upload-item">
              <label for="graph-file-input" class="upload-button">Graph</label>
              <input type="file" id="graph-file-input" accept=".json">
              <span class="file-info" id="graph-file-info">No .json file selected</span>
            </div>
            
            <div class="upload-item">
              <label for="weight-nchw-file-input" class="upload-button">Weights (nchw)</label>
              <input type="file" id="weight-nchw-file-input" accept=".json">
              <span class="file-info" id="weight-nchw-file-info">No .json file selected</span>
            </div>

            <div class="upload-item">
              <label for="weight-nhwc-file-input" class="upload-button">Weights (nhwc)</label>
              <input type="file" id="weight-nhwc-file-input" accept=".json">
              <span class="file-info" id="weight-nhwc-file-info">No .json file selected</span>
            </div>
          </div>
        </div>
        
        <div class="step step-3 disabled">
          <div class="step-icon">3</div>
          <div id="step-3" class="upload-item generate-action">
            <button id="generate-btn" type="button" disabled>${codeLogo} Generate WebNN Code</button>
          </div>
        </div>

        <div class="step step-4 disabled">
          <div class="step-icon">4</div>
          <div id="step-4" class="upload-item download-action">
            <button id="download-btn" type="button" disabled>${downloadLogo} Download Code Files</button>
          </div>
        </div>
      </div>
      <div id="free-dimension-overrides" class="override none"></div>
      <div class="output-panel">
        <div class="left-panel">
          <div class="graph-weight-panel">
            <div id="output-graph" class="panel"></div>
            <div id="output-weight" class="panel"></div>
          </div>
          <div id="log-console" class="status panel"></div>
        </div>
        <div id="output-code"></div>
      </div>
      
      <div class="app-description">
        Disclamer: All model conversion and code generation processes execute entirely within your browser, ensuring your intellectual property remains private and secure as no model data is transmitted to or stored on cloud servers.
      </div>
      <footer>
        <div id="copyright">
          &copy;2025 <a href="https://ibelem.github.io/webnn-code-generator/" title="WebNN Code Generator">WebNN Code Generator</a> · <a href="https://github.com/ibelem/webnn-code-generator/issues" title="WebNN Code Generator Issues">${githubLogo}</a>
        </div>
        <div class="footer-link">
          <a href="http://ibelem.github.io/webnn-code-generator/?graph=https://ibelem.github.io/webnn-code-generator/model/mobilenetv2-12-static/graph.json&weights=https://ibelem.github.io/webnn-code-generator/model/mobilenetv2-12-static/weights.json">Example</a> · <a href="https://ibelem.github.io/netron/" title="WebNN Netron">WebNN Netron</a> · <a href="https://ibelem.github.io/netron/reader.html" title="WebNN Netron">Bin Reader</a> · <a href="https://github.com/huningxin/onnx2webnn" title="Exports the ONNX file to a WebNN JavaScript file and a bin file containing the weights">ONNX2WebNN</a>
        </div>
      </footer>
    </div>
  `;
};

// Initialize the application when the DOM is ready
document.addEventListener('DOMContentLoaded', initializeApp);