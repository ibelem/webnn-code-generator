import { defineConfig } from 'vite';
import monacoEditorEsmPlugin from 'vite-plugin-monaco-editor-esm'

export default defineConfig({
  base: '/webnn-code-generator/',
  plugins: [monacoEditorEsmPlugin()],
})