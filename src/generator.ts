/**
 * WebNN Code Generator
 * Handles the code generation process from model files
 */
import { getModelState, appendLogMessage } from './ui';
import { generateJS } from './code';

// Store freeDims globally for access in generateWebNNCode
let freeDims: string[] = [];

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
  const code = generateJS();
  outputElement.innerHTML = `<pre><code></code></pre>`;
  const codeBlock = outputElement.querySelector('code');
  if (codeBlock) {
    codeBlock.textContent = code; // Use textContent to preserve formatting
  }
}