/**
 * Main entry point for WebNN Code Generator
 */
import './style.css';
import webnnLogo from '/logo/webnn.svg';
import { setupGenerator } from './generator';
import { initializeUI } from './ui';

/**
 * Initialize the application
 */
const initializeApp = (): void => {
  renderAppTemplate();
  initializeUI();
  setupGenerator(document.querySelector<HTMLButtonElement>('#generator'));
};

/**
 * Render the main application template
 */
const renderAppTemplate = (): void => {
  const appContainer = document.querySelector<HTMLDivElement>('#app');
  if (!appContainer) return;

  appContainer.innerHTML = `
    <div class="container">
      <header>
        <a href="https://github.com/ibelem/webnn-code-generator" class="logo-link">
          <img src="${webnnLogo}" class="logo" alt="WebNN logo" />
        </a>
        <h1>WebNN Code Generator</h1>
      </header>
      
      <div class="upload-container">
        <div class="file">
          <label for="graph-upload" class="file-label">Choose Graph/Node File</label>
          <input type="file" id="graph-upload" accept=".json">
          <span class="file-name" id="graph-name">No .json file selected</span>
        </div>
        
        <div class="file">
          <label for="weight-upload" class="file-label">Choose Weight/Bias File</label>
          <input type="file" id="weight-upload" accept=".json">
          <span class="file-name" id="weight-name">No .json file selected</span>
        </div>
        
        <div class="file">
          <label for="bin-upload" class="file-label">Choose BIN File</label>
          <input type="file" id="bin-upload" accept=".bin">
          <span class="file-name" id="bin-name">No .bin file selected</span>
        </div>
        
        <div class="file generator">
          <button id="generator" type="button" disabled>Generate WebNN Code</button>
        </div>
      </div>
      
      <div id="code" class="code"></div>
      <div id="status" class="status"></div>
      
      <div class="description">
        <ul>
          <li>Generate Web Neural Network (WebNN) API vanilla JavaScript code from ONNX, TensorFlow Lite, or other models.</li>
          <li>Model conversion and code generation occur entirely on your local machine, ensuring that none of your model information is stored by this service.</li>
        </ul>
      </div>
      
      <footer>
        &copy;2025 WebNN Code Generator
      </footer>
    </div>
  `;
};

// Initialize the application when the DOM is ready
document.addEventListener('DOMContentLoaded', initializeApp);