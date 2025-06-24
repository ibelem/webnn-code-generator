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
          <div id="step-1" title="Convert your ONNX, TensorFlow Lite, or other model formats into graph (.json) and weights (.bin) files">
            Donwload graph &amp; weights from <a href="https://ibelem.github.io/netron">WebNN Netron</a>
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
              <label for="weight-nchw-bin-file-input" class="upload-button">Weights NCHW</label>
              <input type="file" id="weight-nchw-bin-file-input" accept=".bin">
              <span class="file-info" id="weight-nchw-bin-file-info">No .bin file selected</span>
            </div>
            <div class="upload-item">
              <label for="weight-nhwc-bin-file-input" class="upload-button">Weights NHWC</label>
              <input type="file" id="weight-nhwc-bin-file-input" accept=".bin">
              <span class="file-info" id="weight-nhwc-bin-file-info">No .bin file selected</span>
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
          <div class="example-wrapper" style="position: relative;">
            <span class="pink-link" id="example-menu-trigger">Example</span>
            <div id="example-menu" class="example-menu">
              <div class="example-menu-content">
                <div>
                  <div class="example-title">MobileNet V2</div>
                    <div class="example-links">
                      <a href="https://ibelem.github.io/webnn-code-generator/?graph=https://ibelem.github.io/webnn-code-generator/model/mobilenetv2-12-static/graph.json&weights_nchw=https://ibelem.github.io/webnn-code-generator/model/mobilenetv2-12-static/weights_nchw.bin&weights_nhwc=https://ibelem.github.io/webnn-code-generator/model/mobilenetv2-12-static/weights_nhwc.bin">Code Generation</a>
                      <a href="https://ibelem.github.io/webnn-code-generator/model/mobilenetv2-12-static/index.html?devicetype=gpu&layout=nchw&run=50">NCHW Test Page</a>
                      <a href="https://ibelem.github.io/webnn-code-generator/model/mobilenetv2-12-static/index.html?devicetype=cpu&layout=nhwc&run=50">NHWC Test Page</a>
                    </div>
                  </div>
                  <div>
                    <div class="example-title">Selfie Segmenter Landscape</div>
                    <div class="example-links">
                      <a href="https://ibelem.github.io/webnn-code-generator/?graph=https://ibelem.github.io/webnn-code-generator/model/selfie_segmenter_landscape_19/graph.json&weights_nchw=https://ibelem.github.io/webnn-code-generator/model/selfie_segmenter_landscape_19/weights_nchw.bin&weights_nhwc=https://ibelem.github.io/webnn-code-generator/model/selfie_segmenter_landscape_19/weights_nhwc.bin">Code Generation</a>
                      <a href="https://ibelem.github.io/webnn-code-generator/model/selfie_segmenter_landscape_19/index.html?devicetype=gpu&layout=nchw&run=50">NCHW Test Page</a>
                      <a href="https://ibelem.github.io/webnn-code-generator/model/selfie_segmenter_landscape_19/index.html?devicetype=cpu&layout=nhwc&run=50">NHWC Test Page</a>
                    </div>
                  </div>
              </div>
            </div>
          </div>
          · <a href="https://ibelem.github.io/netron/" title="WebNN Netron">WebNN Netron</a> · <a href="https://ibelem.github.io/netron/reader.html" title="WebNN Netron">Bin Reader</a> · <a href="https://github.com/huningxin/onnx2webnn" title="Exports the ONNX file to a WebNN JavaScript file and a bin file containing the weights">ONNX2WebNN</a>
        </div>
      </footer>
    </div>
  `;

  setTimeout(() => {
    const trigger = document.getElementById('example-menu-trigger');
    const menu = document.getElementById('example-menu');
    if (!trigger || !menu) return;

    // Position menu properly relative to the trigger
    function positionMenu() {
      const rect = trigger?.getBoundingClientRect();
      const menuWidth = 340; // This should match your min-width in CSS

      if (!rect) return;

      // Check if we're too close to the right edge
      const rightEdgeDistance = window.innerWidth - rect.right;
      const bottomEdgeDistance = window.innerHeight - rect.bottom;
      
      // Default position (above and right-aligned)
      if (menu) {
        menu.style.bottom = (rect.height + 8) + 'px';
        menu.style.right = '0px';
      }
      
      // If we're too close to the right edge, align left instead
      if (rightEdgeDistance < menuWidth && rect.left > menuWidth && menu) {
        menu.style.right = '0px';
      }
      
      // If we're too close to the bottom, show above instead of below
      if (bottomEdgeDistance < 200 && rect.top > 250 && menu) {
        menu.style.bottom = (rect.height + 8) + 'px'; 
        menu.style.top = 'auto';
      } else if (menu) {
        menu.style.top = (rect.height + 8) + 'px';
        menu.style.bottom = 'auto';
      }
    }

    // Show/hide functions
    function showMenu() {
      positionMenu();
      if (menu) {
        menu.style.display = 'block';
      }
    }
    
    function hideMenu() {
      if (menu) {
        menu.style.display = 'none';
      }
    }

    // Event listeners for interactions
    let timeoutId: number;
    
    trigger.addEventListener('mouseenter', () => {
      clearTimeout(timeoutId);
      showMenu();
    });
    
    trigger.addEventListener('mouseleave', () => {
      timeoutId = window.setTimeout(hideMenu, 300);
    });
    
    menu.addEventListener('mouseenter', () => {
      clearTimeout(timeoutId);
    });
    
    menu.addEventListener('mouseleave', () => {
      timeoutId = window.setTimeout(hideMenu, 300);
    });

    // Hide menu when clicking elsewhere
    document.addEventListener('click', (e) => {
      if (!menu.contains(e.target as Node) && e.target !== trigger) {
        hideMenu();
      }
    });
  }, 0);
 
};

// Initialize the application when the DOM is ready
document.addEventListener('DOMContentLoaded', initializeApp);