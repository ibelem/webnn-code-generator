// Import all operation modules dynamically (Vite, Webpack, or Node >=18 with import.meta.glob)
const modules = import.meta.glob('./operation/**/*.ts', { eager: true });

// Build the opHandlers map
const opHandlers: Record<string, (node: any, toJsVarName: (name: string) => string, options?: { nhwc?: boolean, weightModelData?: any }) => string> = {};

for (const path in modules) {
  const mod = modules[path] as Record<string, any>;
  for (const exportName in mod) {
    if (typeof mod[exportName] === 'function') {
      opHandlers[exportName] = mod[exportName];
    }
  }
}

// Export the handlers
export { opHandlers };