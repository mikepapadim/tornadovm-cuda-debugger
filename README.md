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
| `batch` | runs to a breakpoint and prints locals (with their Java meaning), `__shared__`, array arguments and the locals that differ between threads (`--json`) |
| `memcheck` | compute-sanitizer (`--tool memcheck\|racecheck\|initcheck\|synccheck`), with reports mapped to generated lines and grouped. Exits 1 on errors (CI); `--json` gives a summary |
| `ui` | web debugger on `127.0.0.1:7777`: breakpoints, F5/F10/F11, thread focus, locals, shared memory, arrays, warps |
| `dap` | Debug Adapter Protocol server, used by the [VS Code extension](vscode/) (and IntelliJ + LSP4IJ) |
| `launch` / `attach PID` | start a long-running app with debug kernels, then attach to it (and detach) whenever you like; `batch --pid PID` takes a snapshot |
| `diff A.json B.json` | compare two `batch --json` reports thread by thread (the same kernel with different inputs, or buggy against fixed) |

**Breakpoints:** `-b K`, `-b K:L`, `-b K:L@B:T` (only block B, thread T), `-b "K:L if i_10 == 8"` (a condition evaluated per GPU thread).
Inside cuda-gdb: `tcd-break`, `tcd-list`, `tcd-array arg1 float 0 16`, `tcd-shared adf_2 float`.

**Readable locals:** tcd recognises TornadoVM's code patterns, so `f_8 = 67` is shown as `arg1[ctx.globalIdx]` and
`i_9` as `ctx.localIdx`. `argN` is the N-th Java parameter counting from 0, so with a `KernelContext` first,
the first array is `arg1`.

**Speed:** a debug session starts in about 2 s. tcd's agent skips TornadoVM's CUDA transfer warm-up, about 5,000 driver
calls that are free natively but take about 20 s under cuda-gdb (`--no-fast-start` keeps it).

## Use it on your own TornadoVM kernel

Given a task in your code, for example:

```java
public static void vectorAdd(KernelContext ctx, FloatArray a, FloatArray b, FloatArray c) {
    int i = ctx.globalIdx;
    c.set(i, a.get(i) + b.get(i));
}
...
new TaskGraph("s0").task("t0", MyApp::vectorAdd, ctx, a, b, c) ...
```

1. **Build your app as usual**, whether with Maven, Gradle or `javac`. Nothing in your code changes, and the tornado jars come from the SDK.
2. **The kernel name is the Java method name:** `vectorAdd`. Parameters become `arg0`, `arg1`, … by position,
   so `a` is `arg1` here, because `ctx` is `arg0`.
3. **Check for memory errors and races across the whole grid:**
   ```bash
   bin/tcd memcheck -- -cp target/classes com.acme.MyApp               # out-of-bounds, bad addresses
   bin/tcd memcheck --tool racecheck -- -cp target/classes com.acme.MyApp  # shared-memory races
   ```
   Every report names a generated line (`>> vectorAdd:17  ...`) and the source file for it.
4. **Stop at the kernel entry to see the generated CUDA C and its line numbers:**
   ```bash
   bin/tcd run -b vectorAdd -- -cp target/classes com.acme.MyApp
   (cuda-gdb) tcd-list                          # generated source, current line marked
   (cuda-gdb) cuda block (3,0,0) thread (7,0,0) # focus on any GPU thread
   (cuda-gdb) next                              # step
   (cuda-gdb) info locals                       # i_3 is ctx.globalIdx, f_8 is a.get(i), ...
   (cuda-gdb) tcd-array arg1 float 0 8          # contents of FloatArray a
   ```
5. **Script it, or hand it to an agent:** `bin/tcd batch -b vectorAdd:17 --at 3:7 --json -- -cp target/classes com.acme.MyApp`.
   Or use `bin/tcd ui …` and click line numbers to set breakpoints.

Use a small problem size, because debug builds (`-G`) are slow.
Installing with jbang: `jbang app install tcd@mikepapadim/tornadovm-cuda-debugger` (while the repo is private, this needs access to it). Application arguments go after the main
class, as usual. `@Parallel` loop kernels work the same way (the generated code is a grid-stride loop).

**Examples:** a missing barrier and a silent out-of-bounds write, each found and fixed step
by step. See [docs/examples.md](docs/examples.md).

## IDE and CI

- **VS Code:** `vscode/` holds a small extension that runs `tcd dap`. Add a `tornadovm-cuda` launch configuration
  with `"args": ["-cp", "target/classes", "com.acme.MyApp"], "stopOnKernel": ["vectorAdd"]`. The generated
  CUDA C opens at the kernel entry. Set breakpoints in the gutter (conditions work too) and switch GPU thread
  from the Debug Console with `cuda block (2,0,0) thread (5,0,0)`. For IntelliJ, point LSP4IJ's DAP client at `bin/tcd dap`.
- **CI** (self-hosted GPU runner): `bin/tcd memcheck --tool racecheck --tool memcheck --json -- -cp … my.Tests`
  exits 1 on any sanitizer error, and its JSON lists every failing kernel line.

## How it works

- Kernels are compiled with `-G -lineinfo` through `tornado.cuda.compiler.flags`, with the cubin cache disabled.
- The generated sources are dumped, and each device stop maps `tornado_kernel.cu` (the name NVRTC gives every kernel) to the right kernel.
- `gdb/tornado.gdbinit` passes through the signals HotSpot uses internally, and `tcd-break` only ever fires in device code.
- `tcd ui` and `tcd dap` drive `cuda-gdb --interpreter=mi3`. The UI is served by the JDK's HTTP server. There are no dependencies beyond a JDK.
- The start-up agent (`agent/TcdAgent.java`) is compiled on first use against the SDK's ASM. For `tcd launch` it also calls
  `prctl(PR_SET_PTRACER)` so cuda-gdb can attach under Yama `ptrace_scope=1`.

Conditional breakpoints are evaluated by cuda-gdb for every warp that reaches the line: about 35 ms per block
before the target on an RTX 4090. A thread near the end of a large grid can take minutes to reach, so use a small problem size.

## Claude Code skill

```bash
mkdir -p ~/.claude/skills/tornadovm-cuda-debug && cp skill/SKILL.md ~/.claude/skills/tornadovm-cuda-debug/
```

Tested with cuda-gdb 12.6, TornadoVM 7.0.1 (JDK 21) and an RTX 4090. Apache 2.0.
