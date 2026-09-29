# TornadoVM CUDA Debugger for VS Code

A thin client for `tcd dap`. Install it with:

```bash
cd vscode && npx @vscode/vsce package && code --install-extension tornadovm-cuda-debug-*.vsix
```

Or, for development, open this folder in VS Code and press F5.

Set `tornadovmCuda.tcdPath` to `<checkout>/bin/tcd`, make sure `TORNADOVM_HOME` is set, and add:

```json
{ "type": "tornadovm-cuda", "request": "launch", "name": "Debug kernel",
  "args": ["-cp", "target/classes", "com.acme.MyApp"], "stopOnKernel": ["vectorAdd"] }
```

At the kernel entry, VS Code opens the generated CUDA C. Click in the gutter to add breakpoints
(conditions are evaluated per GPU thread). To switch GPU thread, type
`cuda block (2,0,0) thread (5,0,0)` in the Debug Console.
