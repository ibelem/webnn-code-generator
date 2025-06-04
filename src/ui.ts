import internetLogo from '/logo/internet.svg?raw';
import localLogo from '/logo/local.svg?raw';
import * as monaco from 'monaco-editor';
/**
 * UI state management and file handling for WebNN Code Generator
 */

// Application state
interface ModelState {
  graphModelData: Record<string, any> | null;
  weightModelData: Record<string, any> | null
}

// Initialize application state
const modelFileState: ModelState = {
  graphModelData: null,
  weightModelData: null
};

// File upload tracking
let inputSetupComplete = false;

/**
 * Add a log message to the console display
 * @param message - Message to display
 * @param isError - Whether this is an error message
 */
export const appendLogMessage = (message: string, isError = false): void => {
  const logConsole = document.querySelector<HTMLDivElement>('#log-console');
  if (!logConsole) return;

  const styleClass = isError ? 'log-error' : 'log-success';
  const timestamp = new Date().toLocaleTimeString('en-US', { hour12: false }); // Use 24-hour format
  logConsole.innerHTML += `<p class="${styleClass}">[${timestamp}] ${message}</p>`;

  // Scroll to the bottom of the log console
  logConsole.scrollTop = logConsole.scrollHeight;
};

/**
 * Enable/disable the generate button based on whether all required files are loaded
 */
export const updateGenerateButtonState = (): void => {
  const generateBtn = document.querySelector<HTMLButtonElement>('#generate-btn');
  const generateDiv = document.querySelectorAll<HTMLDivElement>('.step-3')[0];
  if (!generateBtn) return;
  const state = !(modelFileState.graphModelData && modelFileState.weightModelData);
  generateDiv?.classList.toggle('disabled', state);
  generateBtn.disabled = state;
};

/**
 * Register a file upload handler for the specified input element
 * @param inputId - ID of the file input element
 * @param callback - Function to call when a file is selected
 */
const setupFileInput = (inputId: string, callback: (file: File) => void): void => {
  const inputElement = document.querySelector<HTMLInputElement>(`#${inputId}`);
  if (!inputElement) return;
  
  inputElement.addEventListener('change', (event) => {
    updateDownloadButtonState(true);
    const files = (event.target as HTMLInputElement).files;
    const file = files && files[0];
    if (file) {
      callback(file);
    }
  });
};

/**
 * Read a file and process its contents
 * @param file - File to read
 * @param callback - Function to call with file contents
 */
const processFileContent = (file: File, callback: (data: any) => void): void => {
  const reader = new FileReader();
  
  reader.onload = () => {
    callback(reader.result);
  };
  
  if (file.type === 'application/json') {
    reader.readAsText(file);
  } else {
    reader.readAsArrayBuffer(file);
  }
};

/**
 * Fetch files from URL parameters if provided
 */
export const fetchFilesFromUrl = async (): Promise<void> => {
  const params = new URLSearchParams(window.location.search);
  const graphUrl = params.get('graph');
  const weightUrl = params.get('weights');

  if (graphUrl && weightUrl) {
    appendLogMessage('Fetching model graph and weights files from URL...');
    try {
      // Fetch all files in parallel and get their responses
      const [graphRes, weightRes] = await Promise.all([
        fetch(graphUrl),
        fetch(weightUrl)
      ]);

      // Parse contents
      const [graphResponse, weightResponse] = await Promise.all([
        graphRes.json(),
        weightRes.json()
      ]);

      // Update file info with size and name
      const updateRemoteFileInfo = (elementId: string, res: Response, url: string, fallbackSize: number) => {
        const element = document.querySelector<HTMLSpanElement>(`#${elementId}`);
        if (!element) return;
        const fileName = url.split('/').pop() || '';
        // Try to get size from header, fallback to content length
        let size = Number(res.headers.get('content-length')) || fallbackSize;
        const fileSizeInKB = size / 1024;
        const fileSize = fileSizeInKB < 1024
          ? `${fileSizeInKB.toFixed(2)} KB`
          : `${(fileSizeInKB / 1024).toFixed(2)} MB`;
        element.innerHTML = `${internetLogo} ${fileName} · ${fileSize}`;
      };

      updateRemoteFileInfo('graph-file-info', graphRes, graphUrl, JSON.stringify(graphResponse).length);
      updateRemoteFileInfo('weight-file-info', weightRes, weightUrl, JSON.stringify(weightResponse).length);

      modelFileState.graphModelData = graphResponse;
      modelFileState.weightModelData = weightResponse;

      appendLogMessage('Model graph and weights files fetched successfully.');
      renderGraphDetails(modelFileState.graphModelData?.graph[0]); // Render graph details
      if (modelFileState.weightModelData) {
        renderWeightDetails(modelFileState.weightModelData as Record<string, any>); // Render weight details
      }
      updateStep1State(false);
      updateStep2State(false);
      updateGenerateButtonState();
    } catch (error) {
      console.error('Error fetching files from URL:', error);
      appendLogMessage('Failed to fetch files from URL.', true);
    }
  }
};

/**
 * Enable/disable the Step 1
 */
export const updateStep1State = (disabled:boolean): void => {
  const step1Div = document.querySelectorAll<HTMLDivElement>('.step-1')[0];
  (disabled === false) ? step1Div?.classList.remove('disabled') : step1Div?.classList.add('disabled');
};

/**
 * Enable/disable the Step 2
 */
export const updateStep2State = (disabled:boolean): void => {
  const step1Div = document.querySelectorAll<HTMLDivElement>('.step-2')[0];
  (disabled === false) ? step1Div?.classList.remove('disabled') : step1Div?.classList.add('disabled');
};

/**
 * Set up file upload handlers for all file inputs
 */
export const setupFileInputs = (): void => {
  // Only set up handlers once
  if (inputSetupComplete) return;
  
  // Graph file upload
  setupFileInput('graph-file-input', (file) => {
    processFileContent(file, (data) => {
      try {
        modelFileState.graphModelData = JSON.parse(data as string);
        updateFileInfo('graph-file-info', file);
        appendLogMessage('Graph file loaded successfully');
        renderGraphDetails(modelFileState.graphModelData?.graph[0]); // Render graph details
        updateGenerateButtonState();
      } catch (error) {
        appendLogMessage('Error parsing graph file: ' + (error as Error).message, true);
      }
    });
  });

  // Weight file upload
  setupFileInput('weight-file-input', (file) => {
    processFileContent(file, (data) => {
      try {
        modelFileState.weightModelData = JSON.parse(data as string);
        updateFileInfo('weight-file-info', file);
        appendLogMessage('Weight file loaded successfully');
        if (modelFileState.weightModelData) {
          renderWeightDetails(modelFileState.weightModelData as Record<string, any>); // Render weight details
        }
        updateStep1State(false);
        updateStep2State(false);
        updateGenerateButtonState();
      } catch (error) {
        appendLogMessage('Error parsing weight file: ' + (error as Error).message, true);
      }
    });
  });
  
  inputSetupComplete = true;
};

/**
 * Update the displayed file information (size and name)
 * @param elementId - ID of the element to update
 * @param file - The file object
 */
const updateFileInfo = (elementId: string, file: File): void => {
  const element = document.querySelector<HTMLSpanElement>(`#${elementId}`);
  if (!element) return;

  // Determine file size in KB or MB
  const fileSizeInKB = file.size / 1024;
  const fileSize = fileSizeInKB < 1024
    ? `${fileSizeInKB.toFixed(2)} KB`
    : `${(fileSizeInKB / 1024).toFixed(2)} MB`;

  // Update the element with file size and name
  element.innerHTML = `${localLogo} ${file.name} · ${fileSize}`;
};

/**
 * Initialize the UI
 */
let monacoEditor: monaco.editor.IStandaloneCodeEditor | null = null;

export const initializeInterface = (): void => {
  updateGenerateButtonState();
  fetchFilesFromUrl();
  setupFileInputs();
  const outputElement = document.querySelector<HTMLDivElement>('#output-code');
  if (!outputElement) return;
  monacoEditor = monaco.editor.create(outputElement, {
    value: '// WebNN Code Generator',
    language: 'javascript',
    fontSize: 12,
    fontFamily: 'Intel One Mono'
  });

  // Resize Monaco editor on window resize
  window.addEventListener('resize', () => {
    if (monacoEditor) {
      monacoEditor.layout();
    }
  });

  // Add click handler for download button
  const downloadBtn = document.querySelector<HTMLButtonElement>('#download-btn');
  if (downloadBtn) {
    downloadBtn.addEventListener('click', () => {
      import('./code').then(mod => {
        mod.downloadJS();
        mod.downloadHTML();
      });
    });
  }
};

/**
 * Get the application state
 * @returns The current application state
 */
export const getModelState = (): ModelState => modelFileState;

/**
 * Render graph details (inputs, outputs, nodes) in the output-graph element
 * @param graphData - The graph data from the uploaded or fetched file
 */
const renderGraphDetails = (graphData: any): void => {
  const outputGraphElement = document.querySelector<HTMLDivElement>('#output-graph');
  const overrideDiv = document.getElementById('free-dimension-overrides');
  if (!outputGraphElement || !overrideDiv) return;

  outputGraphElement.innerHTML = '';
  overrideDiv.innerHTML = '';
  overrideDiv.className = 'override none';

  // Helper to collect all string dimensions
  const freeDims: Set<string> = new Set();

  // Render graph inputs
  if (graphData.inputs) {
    const inputsHTML = graphData.inputs.map((input: any) => {
      const dims = input.value?.[0]?.type?.shape?.dimensions || [];
      dims.forEach((dim: string | number) => {
        if (typeof dim === 'string') freeDims.add(dim);
      });
      return `
        <div class="graph-section">
          <span class="inputs" title="Inputs">I</span>
          <span class="name" title="${input.name}">${input.name}</span>
          <span></span>
          <span class="tensor">
            ${input.value[0]?.type?.dataType || ''}
            ${getShapeString(dims)}
          </span>
        </div>
      `;
    }).join('');
    outputGraphElement.innerHTML += `<div class="graph-inputs">${inputsHTML}</div>`;
  }

  // Render graph outputs (unchanged)
  if (graphData.outputs) {
    const outputsHTML = graphData.outputs.map((output: any) => `
      <div class="graph-section">
        <span class="outputs" title="Outputs">O</span>
        <span class="name" title="${output.name}">${output.name}</span>
        <span></span>
        <span class="tensor" title="${output.value[0]?.type?.dataType || ''} ${getShapeString(output.value[0]?.type?.shape?.dimensions)}">
          ${output.value[0]?.type?.dataType || ''}
          ${getShapeString(output.value[0]?.type?.shape?.dimensions)}
        </span>
      </div>
    `).join('');
    outputGraphElement.innerHTML += `<div class="graph-outputs">${outputsHTML}</div>`;
  }

  // Render graph nodes (unchanged)
  if (graphData.nodes) {
    const nodesHTML = graphData.nodes.map((node: any) => `
      <div class="node-inputs-outputs">
        <div class="type" title="${node.type?.name || ''}">${node.type?.name || ''}</div>
        <div class="pink name" title="${node.name || node.identifier}">${node.name || node.identifier}</div>
        <div class="inputs" title="Inputs">I</div>
        <div>
          ${node.inputs.map((input: any) => `
            <div class="initializer">
              <span class="inputoutput" title="${input.name}">${input.name}</span> 
              <span class="green name" title="${(input.value[0]?.initializer?.name || input.value[0]?.initializer?.identifier) ?? input.value[0]?.name ?? ''}">${(input.value[0]?.initializer?.name || input.value[0]?.initializer?.identifier) ?? input.value[0]?.name ?? ''}</span> 
              <span></span>
              <span class="tensor" title="${input.value[0]?.type?.dataType || ''}${getShapeString(input.value[0]?.type?.shape?.dimensions)}">
                ${input.value[0]?.type?.dataType || ''}${getShapeString(input.value[0]?.type?.shape?.dimensions)}
              </span>
            </div>
          `).join('')}
        </div>
        <div class="outputs" title="Outputs">O</div>
        <div>
          ${node.outputs.map((output: any) => `
            <div class="initializer">
              <span class="inputoutput" title="${output.name}">${output.name}</span> <span class="name" title="${output.value[0]?.name || ''}">${output.value[0]?.name || ''}</span>
            </div>
          `).join('')}
        </div>
      </div>
    `).join('');
    outputGraphElement.innerHTML += `<div class="graph-nodes">${nodesHTML}</div>`;
  }



  // Render free dimension overrides if needed
  if (freeDims.size > 0) {
    overrideDiv.className = 'override';
    overrideDiv.innerHTML = `<div class="override-config">
      Set <a href="https://webnn.io/en/learn/tutorials/onnx-runtime/free-dimension-overrides">free dimension overrides</a>: 
      ${Array.from(freeDims).map(dim => `
        <span>${dim}</span> <input type="text" id="override_${dim}" name="override_${dim}" required size="5" />
      `).join(' ')}
      </div>
    `;
    // Pass freeDims to generator for validation
    import('./ui').then(mod => {
      if (mod.setFreeDims) mod.setFreeDims(Array.from(freeDims));
    });
  } else {
    overrideDiv.className = 'override';
    import('./ui').then(mod => {
      if (mod.setFreeDims) mod.setFreeDims([]);
    });
  }
};

/**
 * Render weight details (node type, node name, inputs) in the output-weight element
 * @param weightData - The weight data from the uploaded or fetched file
 */
const renderWeightDetails = (weightData: Record<string, any>): void => {
  const outputWeightElement = document.querySelector<HTMLDivElement>('#output-weight');
  if (!outputWeightElement) return;

  // Convert weightData to array and sort by nodeIdentifier if present
  const nodes = Object.values(weightData)
    .filter(node => node && node.nodeIdentifier !== undefined)
    .sort((a, b) => Number(a.nodeIdentifier) - Number(b.nodeIdentifier));

  // If some nodes don't have nodeIdentifier, append them at the end
  const noIdNodes = Object.values(weightData)
    .filter(node => node && node.nodeIdentifier === undefined);

  const orderedNodes = [...nodes, ...noIdNodes];

  const nodesHTML = orderedNodes.map((node) => {
    return `
      <div class="weight-section">
        <span class="type" title="${node.nodeType || ''}">${node.nodeType || ''}</span>
        <span class="pink nodename" title="${node.nodeName || node.nodeIdentifier}">${node.nodeName || node.nodeIdentifier}</span>
        <span class="inputoutput" title="${node.input || ''}">${node.input || ''}</span>
        <span class="green name" title="${node.name || ''}">${node.name || ''}</span>
        <span></span>
        <span class="tensor" title="${node.dataType || ''}[${node.shape?.join(', ') || ''}]">${node.dataType || ''}[${node.shape?.join(', ') || ''}]</span>
      </div>
    `;
  }).join('');

  outputWeightElement.innerHTML = nodesHTML;
};

const getShapeString = (dims?: number[]) => {
  if (Array.isArray(dims) && dims.length > 0) {
    return `[${dims.join(',')}]`;
  }
  return '';
};

// Store freeDims globally for access in generateWebNNCode
let freeDims: string[] = [];
    // Initial freeDimsOverrides object
export let freeDimsOverrides: Record<string, number | null> = {};

/**
 * Set free dimension names for validation before code generation
 */
export function setFreeDims(dims: string[]) {
  freeDims = dims;
}

/**
 * Set up the generator button with event listeners
 * @param button - The generator button element
 */
export function initializeCodeGenerator(button: HTMLButtonElement | null): void {
  if (!button) {
    console.error('Generator button not found');
    return;
  }

  button.addEventListener('click', () => {
    updateDownloadButtonState(true);
    // Check for freeDims and their input values before generating code
    freeDims.forEach(dim => {
      freeDimsOverrides[dim] = null;
    });

    let missing = false;
    freeDims.forEach(dim => {
      const input = document.getElementById(`override_${dim}`) as HTMLInputElement | null;
      const value = input?.value.trim();
      if (!input || value === '') {
        appendLogMessage(`You need to set free dimension override for "${dim}" before generating WebNN code`, true);
        missing = true;
      } else if (isNaN(Number(value))) {
        appendLogMessage(`The free dimension override value for "${dim}" must be a number`, true);
        missing = true;
      } else {
        freeDimsOverrides[dim] = Number(value);
      }
    });
    if (missing) return;

    generateWebNNCode();
    updateDownloadButtonState(false);
  });
}

/**
 * Enable/disable the download button based on whether generateWebNNCode() is completed
 * and the generated code is available
 */
export const updateDownloadButtonState = (disabled:boolean): void => {
  const downloadBtn = document.querySelector<HTMLButtonElement>('#download-btn');
  const downloadDiv = document.querySelectorAll<HTMLDivElement>('.step-4')[0];
  if (!downloadBtn) return;
  (disabled === false) ? downloadDiv?.classList.remove('disabled') : downloadDiv?.classList.add('disabled');
  (disabled === false) ? downloadBtn.disabled = false : downloadBtn.disabled = true;
};

/**
 * Generate WebNN code from the loaded model files
 */
function generateWebNNCode(): void {
  monaco.editor.getModels()[0].setValue('Starting code generation process...');
  appendLogMessage('Starting code generation process...');
  
  try {
    const { graphModelData, weightModelData} = getModelState();
    if (!graphModelData || !weightModelData) {
      appendLogMessage('Missing required files for code generation', true);
      return;
    }
    setTimeout(() => {
      renderOutputCode();
      appendLogMessage('Vanilla JavaScript code generation for WebNN completed successfully');
    }, 500);
  } catch (error) {
    console.error('Error during code generation:', error);
    appendLogMessage(`Code generation failed: ${(error as Error).message}`, true);
    monaco.editor.getModels()[0].setValue(`Code generation failed. See status for details.`);
  }
}

/**
 * Display the generated code in the code element
 */
function renderOutputCode(): void {
  const codeTab = document.createElement('div');
  codeTab.className = 'code-tab';
  codeTab.innerHTML = `
    <input type="radio" id="nchw-js" name="tabs" checked />
    <label class="tab" for="nchw-js">nchw.js</label>
    <input type="radio" id="nhwc-js" name="tabs" />
    <label class="tab" for="nhwc-js">nhwc.js</label>
    <input type="radio" id="webnn-html" name="tabs" />
    <label class="tab" for="webnn-html">webnn.html</label>
    <span class="glider"></span>
  `;

  const overrideDiv = document.getElementById('free-dimension-overrides');
  overrideDiv?.appendChild(codeTab);

  import('./code').then(mod => {
    const code = mod.generateJS();
    const html = mod.generateHTML ? mod.generateHTML() : '';

    // Set default to nchw.js
    monaco.editor.getModels()[0].setValue(code.nchw);

    // Add event listeners for tab switching
    document.getElementById('nchw-js')?.addEventListener('change', function () {
      if ((this as HTMLInputElement).checked) {
        monaco.editor.getModels()[0].setValue(code.nchw);
        monaco.editor.setModelLanguage(monaco.editor.getModels()[0], 'javascript');
      }
    });
    document.getElementById('nhwc-js')?.addEventListener('change', function () {
      if ((this as HTMLInputElement).checked) {
        monaco.editor.getModels()[0].setValue(code.nhwc);
        monaco.editor.setModelLanguage(monaco.editor.getModels()[0], 'javascript');
      }
    });
    document.getElementById('webnn-html')?.addEventListener('change', function () {
      if ((this as HTMLInputElement).checked) {
        monaco.editor.getModels()[0].setValue(html);
        monaco.editor.setModelLanguage(monaco.editor.getModels()[0], 'html');
      }
    });
  });
}