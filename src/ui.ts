/**
 * UI state management and file handling for WebNN Code Generator
 */

// Application state
interface AppState {
  graphData: Record<string, any> | null;
  weightData: Record<string, any> | null;
  binData: ArrayBuffer | null;
}

// Initialize application state
const state: AppState = {
  graphData: null,
  weightData: null,
  binData: null
};

// File upload tracking
let uploadHandlersInitialized = false;

/**
 * Add a status message to the status container
 * @param message - Message to display
 * @param isError - Whether this is an error message
 */
export const updateStatus = (message: string, isError = false): void => {
  const statusElement = document.querySelector<HTMLDivElement>('#status');
  if (!statusElement) return;

  const style = isError ? 'error' : 'success';
  const timestamp = new Date().toLocaleTimeString('en-US', { hour12: false }); // Use 24-hour format
  statusElement.innerHTML += `<p class="${style}">[${timestamp}] ${message}</p>`;

  // Scroll to the bottom of the status element
  statusElement.scrollTop = statusElement.scrollHeight;
};

/**
 * Enable/disable the generate button based on whether all required files are loaded
 */
export const updateGenerateButtonState = (): void => {
  const generateButton = document.querySelector<HTMLButtonElement>('#generator');
  if (!generateButton) return;
  
  generateButton.disabled = !(state.graphData && state.weightData && state.binData);
};

/**
 * Register a file upload handler for the specified input element
 * @param inputId - ID of the file input element
 * @param callback - Function to call when a file is selected
 */
const registerFileUploadHandler = (inputId: string, callback: (file: File) => void): void => {
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
const readFile = (file: File, callback: (data: any) => void): void => {
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

  if (!(graphUrl && weightUrl && binUrl)) return;

  updateStatus('Fetching files from URL...');
  
  try {
    const [graphResponse, weightResponse, binResponse] = await Promise.all([
      fetch(graphUrl).then(res => res.json()),
      fetch(weightUrl).then(res => res.json()),
      fetch(binUrl).then(res => res.arrayBuffer())
    ]);
    
    state.graphData = graphResponse;
    state.weightData = weightResponse;
    state.binData = binResponse;
    
    updateStatus('Files fetched successfully!');
    updateGenerateButtonState();
  } catch (error) {
    console.error('Error fetching files from URL:', error);
    updateStatus('Failed to fetch files from URL.', true);
  }
};

/**
 * Set up file upload handlers for all file inputs
 */
export const setupFileUploads = (): void => {
  // Only set up handlers once
  if (uploadHandlersInitialized) return;
  
  // Graph file upload
  registerFileUploadHandler('graph-upload', (file) => {
    readFile(file, (data) => {
      try {
        state.graphData = JSON.parse(data as string);
        updateFileName('graph-name', file.name);
        updateStatus('Graph file loaded successfully');
        updateGenerateButtonState();
      } catch (error) {
        updateStatus('Error parsing graph file: ' + (error as Error).message, true);
      }
    });
  });

  // Weight file upload
  registerFileUploadHandler('weight-upload', (file) => {
    readFile(file, (data) => {
      try {
        state.weightData = JSON.parse(data as string);
        updateFileName('weight-name', file.name);
        updateStatus('Weight file loaded successfully');
        updateGenerateButtonState();
      } catch (error) {
        updateStatus('Error parsing weight file: ' + (error as Error).message, true);
      }
    });
  });

  // Binary file upload
  registerFileUploadHandler('bin-upload', (file) => {
    readFile(file, (data) => {
      state.binData = data as ArrayBuffer;
      updateFileName('bin-name', file.name);
      updateStatus('BIN file loaded successfully');
      updateGenerateButtonState();
    });
  });
  
  uploadHandlersInitialized = true;
};

/**
 * Update the displayed filename for an uploaded file
 * @param elementId - ID of the element to update
 * @param fileName - Name of the file
 */
const updateFileName = (elementId: string, fileName: string): void => {
  const element = document.querySelector<HTMLSpanElement>(`#${elementId}`);
  if (element) {
    element.textContent = fileName;
  }
};

/**
 * Initialize the UI
 */
export const initializeUI = (): void => {
  updateGenerateButtonState();
  fetchFilesFromUrl();
  setupFileUploads();
};

/**
 * Get the application state
 * @returns The current application state
 */
export const getAppState = (): AppState => {
  return { ...state };
};