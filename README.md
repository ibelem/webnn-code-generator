# WebNN Code Generator

Visit [https://ibelem.github.io/webnn-code-generator/](https://ibelem.github.io/webnn-code-generator/)

## Overview

The [WebNN Code Generator](https://github.com/ibelem/webnn-code-generator/) offers a user-friendly web interface for generating WebNN-compatible JavaScript code. The tool supports various model formats (ONNX, TensorFlow Lite, or others) and generates optimized code for browser-based machine learning inference.

### 🔒 Client-Side Processing Only

All model conversion and code generation processes execute entirely within your browser, ensuring your intellectual property remains private and secure as no model data is transmitted to or stored on cloud servers.

## Workflow

The conversion process involves two complementary tools:

1. [WebNN Netron](https://ibelem.github.io/netron/) - Extracts model structure and weights
2. [WebNN Code Generator](https://ibelem.github.io/webnn-code-generator/) - Generates WebNN JavaScript code

## Step-by-Step Guide

### Step 1: Extract Model Components

1. Navigate to [WebNN Netron](https://ibelem.github.io/netron/)
2. Click the **"Open Model..."** button
3. Select your model file (`.onnx`, `.tflite`, or other supported formats)
4. Once loaded, download the required files:
   - Click **"Graph"** → Download `graph.json`
   - Click **"Weights"** → Download `weights_nchw.json`, `weights_nchw.bin`, `weights_nhwc.json` and `weights_nhwc.bin`

### Step 2: Generate WebNN Code

1. Open [WebNN Code Generator](https://ibelem.github.io/webnn-code-generator/)
2. Upload the extracted files:
   - Click **"Choose Graph"** → Select `graph.json`
   - Click **"Choose Weights"** → Select `weights_nchw.json` and `weights_nhwc.json`

### Step 3: Configure Dynamic Dimensions (if needed)

If your model contains symbolic dimensions, you'll see a **"Set free dimension overrides"** section:

1. Enter specific values for each dynamic dimension
2. This resolves variables like batch size or input dimensions to concrete values
3. See [symbolic dimensions documentation](https://webnn.io/en/learn/tutorials/onnx-runtime/free-dimension-overrides) for details

### Step 4: Generate and Download

1. Click **"Generate WebNN Code"**
2. Click **"Download Code Files"** to receive:
   - Generated JavaScript file with WebNN implementation
   - `webnn.html` test file for validation
3. Put the `weights_nchw.bin` and `weights_nhwc.bin` downloaded from [WebNN Netron](https://ibelem.github.io/netron/) together with `.js` and `webnn.html` in the same folder

## Testing Your Generated Code

### Security Requirements

WebNN requires a [secure context](https://developer.mozilla.org/en-US/docs/Web/Security/Secure_Contexts) to function. Valid environments include:

- `https://` URLs
- `http://localhost` or `http://127.0.0.1`
- Local development servers

### Local Testing

Start a local HTTP server to test your generated code:

```bash
# Install http-server if needed
npm install -g http-server

# Start server in your project directory
http-server

# Navigate to http://localhost:8080 in your browser
```

Open the generated `webnn.html` file in your browser to validate the conversion.