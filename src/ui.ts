/**
 * UI state management and file handling for WebNN Code Generator
 */

// Application state
interface ModelState {
  graphModelData: Record<string, any> | null;
  weightModelData: Record<string, any> | null;
  binaryModelData: ArrayBuffer | null;
}

// Initialize application state
const modelFileState: ModelState = {
  graphModelData: null,
  weightModelData: null,
  binaryModelData: null
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

  const state = !(modelFileState.graphModelData && modelFileState.weightModelData && modelFileState.binaryModelData);
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
  const weightUrl = params.get('weight');
  const binUrl = params.get('bin');

  if (graphUrl && weightUrl && binUrl) {
    appendLogMessage('Fetching files from URL...');
    try {
      const [graphResponse, weightResponse, binResponse] = await Promise.all([
        fetch(graphUrl).then(res => res.json()),
        fetch(weightUrl).then(res => res.json()),
        fetch(binUrl).then(res => res.arrayBuffer())
      ]);

      modelFileState.graphModelData = graphResponse;
      modelFileState.weightModelData = weightResponse;
      modelFileState.binaryModelData = binResponse;

      appendLogMessage('Files fetched successfully!');
      renderGraphDetails(modelFileState.graphModelData?.graph[0]); // Render graph details
      if (modelFileState.weightModelData) {
        renderWeightDetails(modelFileState.weightModelData as Record<string, any>); // Render weight details
      }
      updateGenerateButtonState();
    } catch (error) {
      console.error('Error fetching files from URL:', error);
      appendLogMessage('Failed to fetch files from URL.', true);
    }
  }
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
        updateGenerateButtonState();
      } catch (error) {
        appendLogMessage('Error parsing weight file: ' + (error as Error).message, true);
      }
    });
  });

  // Binary file upload
  setupFileInput('bin-file-input', (file) => {
    processFileContent(file, (data) => {
      modelFileState.binaryModelData = data as ArrayBuffer;
      updateFileInfo('bin-file-info', file);
      appendLogMessage('BIN file loaded successfully');
      updateGenerateButtonState();
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
  element.textContent = `${fileSize} · ${file.name}`;
};

/**
 * Initialize the UI
 */
export const initializeInterface = (): void => {
  updateGenerateButtonState();
  fetchFilesFromUrl();
  setupFileInputs();
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
        <div class="pink name" title="${node.name || ''}">${node.name || ''}</div>
        <div class="inputs" title="Inputs">I</div>
        <div>
          ${node.inputs.map((input: any) => `
            <div class="initializer">
              <span class="inputoutput" title="${input.name}">${input.name}</span> 
              <span class="green name" title="${input.value[0]?.initializer?.name ?? input.value[0]?.name ?? ''}">${input.value[0]?.initializer?.name ?? input.value[0]?.name ?? ''}</span> 
              <span></span>
              <span class="tensor" title="${input.value[0]?.type?.dataType || ''} ${getShapeString(input.value[0]?.type?.shape?.dimensions)}">
                ${input.value[0]?.type?.dataType || ''}
                ${getShapeString(input.value[0]?.type?.shape?.dimensions)}
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
    overrideDiv.innerHTML = `
      Set <a href="https://webnn.io/en/learn/tutorials/onnx-runtime/free-dimension-overrides">free dimension overrides</a>: 
      ${Array.from(freeDims).map(dim => `
        <span>${dim}</span> <input type="text" id="override_${dim}" name="override_${dim}" required size="5" />
      `).join(' ')}
    `;
    // Pass freeDims to generator for validation
    import('./ui').then(mod => {
      if (mod.setFreeDims) mod.setFreeDims(Array.from(freeDims));
    });
  } else {
    overrideDiv.className = 'override none';
    overrideDiv.innerHTML = '';
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

  // Clear previous content
  outputWeightElement.innerHTML = '';

  // Check if weightData is valid
  if (!weightData || typeof weightData !== 'object') {
    outputWeightElement.innerHTML = '<p>No weight data found in the file.</p>';
    return;
  }

  // Render weight nodes
  const nodesHTML = Object.keys(weightData).map((key) => {
    const node = weightData[key];
    return `
      <div class="weight-section">
        <span class="type" title="${node.nodeType || ''}">${node.nodeType || ''}</span>
        <span class="pink nodename" title="${node.nodeName || ''}">${node.nodeName || ''}</span>
        <span class="inputoutput" title="${node.input || ''}">${node.input || ''}</span>
        <span class="green name" title="${node.name || ''}">${node.name || ''}</span>
        <span></span>
        <span class="tensor" title="${node.dataType || ''}[${node.shape?.join(', ') || ''}]">${node.dataType || ''}[${node.shape?.join(', ') || ''}]</span>
      </div>
    `;
  }).join('');

  outputWeightElement.innerHTML = nodesHTML || '<p>No weight nodes found in the file.</p>';
};

const getShapeString = (dims?: number[]) => {
  if (Array.isArray(dims) && dims.length > 0) {
    return `[${dims.join(',')}]`;
  }
  return '';
};

// Store freeDims globally for access in generateWebNNCode
let freeDims: string[] = [];

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
    // Check for freeDims and their input values before generating code
    if (freeDims.length > 0) {
      let missing = false;
      for (const dim of freeDims) {
        const input = document.getElementById(`override_${dim}`) as HTMLInputElement | null;
        const value = input?.value.trim();
        if (!input || value === '') {
          appendLogMessage(`You need to set free dimension override for "${dim}" before generating WebNN code`, true);
          missing = true;
        } else if (isNaN(Number(value))) {
          appendLogMessage(`The free dimension override value for "${dim}" must be a number`, true);
          missing = true;
        }
      }
      if (missing) return;
    }
    generateWebNNCode();
  });
}

/**
 * Generate WebNN code from the loaded model files
 */
function generateWebNNCode(): void {
  const outputElement = document.querySelector<HTMLDivElement>('#output-code');
  if (!outputElement) return;

  appendLogMessage('Starting code generation process...');
  
  try {
    const { graphModelData, weightModelData, binaryModelData } = getModelState();
    if (!graphModelData || !weightModelData || !binaryModelData) {
      appendLogMessage('Missing required files for code generation', true);
      return;
    }
    outputElement.innerHTML = '<pre><code>WebNN Code Generator is running...</code></pre>';
    setTimeout(() => {
      renderOutputCode(outputElement); // Your code generation logic here
      appendLogMessage('Vanilla JavaScript code generation for WebNN completed successfully');
    }, 500);
  } catch (error) {
    console.error('Error during code generation:', error);
    appendLogMessage(`Code generation failed: ${(error as Error).message}`, true);
    outputElement.innerHTML = '<div class="log-error">Code generation failed. See status for details.</div>';
  }
}

/**
 * Display the generated code in the code element
 * @param outputElement - The element to display code in
 */
function renderOutputCode(outputElement: HTMLDivElement): void {
  // Import generateJS dynamically to avoid circular dependency if needed
  import('./code').then(mod => {
    const code = mod.generateJS();
    outputElement.innerHTML = `<pre><code></code></pre>`;
    const codeBlock = outputElement.querySelector('code');
    if (codeBlock) {
      codeBlock.textContent = code; // Use textContent to preserve formatting
    }
  });
}