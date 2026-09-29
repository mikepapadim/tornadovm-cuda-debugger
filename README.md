<p align="center"><img src="docs/logo.svg" alt="tcd: TornadoVM CUDA debugger" width="360"></p>

Breakpoints, per-thread inspection and sanitizers for the CUDA kernels that
[TornadoVM](https://github.com/beehive-lab/TornadoVM) generates from your Java code. It uses
cuda-gdb and compute-sanitizer underneath, and works with unmodified TornadoVM SDKs that have the CUDA backend.

![tcd web UI stopped in a racy reduction](docs/ui.png)

## Quick start

```bash
export TORNADOVM_HOME=~/.sdkman/candidates/tornadovm/7.0.1-jdk21-cuda
bin/tcd doctor                                                     # check the setup
bin/tcd memcheck --tool racecheck -- -cp classes my.Main           # find the bug
bin/tcd batch -b myKernel:33 --at 0:100 -- -cp classes my.Main     # inspect a thread
bin/tcd ui -b myKernel -- -cp classes my.Main                      # web debugger
```

Put your normal `java` arguments after `--`; tcd adds the TornadoVM flags itself. A kernel is
named after its Java method. Lines are lines of the generated CUDA C. `--at B:T` selects a block and thread
(`2:5` for 1-D, `2,0,0:5,1,0` for 3-D). `bin/tcd` runs `tcd.java` with JBang, or with `java` (JDK 21+).

| command | |
|---|---|
| `doctor` | checks cuda-gdb, the driver, the SDK and permissions |
| `run` | interactive cuda-gdb, set up for the JVM (`--tui` for the TUI) |
| `batch` | runs to a breakpoint and prints locals, `__shared__` and `-p` expressions per thread (`--json`) |
| `memcheck` | compute-sanitizer (`--tool memcheck\|racecheck\|initcheck\|synccheck`), with reports mapped to generated lines and grouped |
| `ui` | web debugger on `127.0.0.1:7777`: breakpoints, F5/F10/F11, thread focus, locals, shared memory, arrays, warps |

Inside cuda-gdb: `tcd-break K[:L]`, `tcd-list`, `tcd-array arg1 float 0 16`, `tcd-shared adf_2 float`.

**Examples:** a missing barrier and a silent out-of-bounds write, each found and fixed step
by step. See [docs/examples.md](docs/examples.md).

## How it works

- Kernels are compiled with `-G -lineinfo` through `tornado.cuda.compiler.flags`, with the cubin cache disabled.
- The generated sources are dumped, and each device stop maps `tornado_kernel.cu` (the name NVRTC gives every kernel) to the right kernel.
- `gdb/tornado.gdbinit` passes through the signals HotSpot uses internally, and `tcd-break` only ever fires in device code.
- `tcd ui` drives `cuda-gdb --interpreter=mi3` behind the JDK's HTTP server. There are no dependencies beyond a JDK.

## Claude Code skill

```bash
mkdir -p ~/.claude/skills/tornadovm-cuda-debug && cp skill/SKILL.md ~/.claude/skills/tornadovm-cuda-debug/
```

Tested with cuda-gdb 12.6, TornadoVM 7.0.1 (JDK 21) and an RTX 4090. Apache 2.0.
