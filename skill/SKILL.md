---
name: tornadovm-cuda-debug
description: Debug TornadoVM CUDA-backend GPU kernels with cuda-gdb and compute-sanitizer through the tcd tool (tornadovm-cuda-debugger). Use when a TornadoVM kernel (TaskGraph task, KernelContext kernel, @Parallel loop) running on the CUDA backend gives wrong results, crashes with CUDA_ERROR_LAUNCH_FAILED / illegal address, hangs, races, or reads out of bounds; when the user wants to step through, breakpoint, or inspect GPU threads, warps, shared memory or TornadoVM arrays of a TornadoVM kernel; or mentions tcd, cuda-gdb, compute-sanitizer, memcheck or racecheck together with TornadoVM.
---

# Debugging TornadoVM CUDA kernels with tcd

`tcd` wraps cuda-gdb and compute-sanitizer for TornadoVM. It works with unmodified TornadoVM
SDKs that have the **CUDA** backend (NVRTC); the older PTX backend is not supported.
Repo: https://github.com/mikepapadim/tornadovm-cuda-debugger. Below, `tcd` means
`<checkout>/bin/tcd` (or `jbang <checkout>/tcd.java`).

## Workflow: locate first, then inspect

1. **Environment.** `export TORNADOVM_HOME=<CUDA SDK>` then `tcd doctor`. Fix every `[FAIL]`
   before going on.
2. **Find the Java command.** Everything after `--` is what you would pass to `java`
   (`-cp classes my.Main args`, or `-jar app.jar`). tcd adds the TornadoVM JVM flags itself.
   Do not pass the `tornado` launcher.
3. **Locate the bug with the sanitizer.** This is cheap, needs no breakpoints, and covers the whole grid:
   - wrong results that vary between runs, or shared memory / `allocate*LocalArray` in use → `tcd memcheck --tool racecheck -- …`
   - crash, `CUDA_ERROR_LAUNCH_FAILED`, illegal address, grid larger than the data → `tcd memcheck -- …`
   - garbage from uninitialised local memory → `--tool initcheck`; a barrier inside divergent code → `--tool synccheck`

   Each report comes with `>> kernel:LINE  <generated code>`, and its path points at
   `~/.tornado-cuda-debug/sessions/<ts>/src/<kernel>.cu`. For memcheck-style reports that name a
   thread, the summary prints a ready-made `tcd batch -b kernel:LINE@B:T` command that stops exactly on that
   thread. Race reports name no thread, so choose them yourself (see step 4).
   For scripts and agents, `--json` prints only the grouped summary. The exit code is 1 when there are sanitizer errors.
4. **Inspect the failing thread.**
   `tcd batch -b KERNEL:LINE --at B:T [--at B:T] [--hits N] [-p expr,...] --json -- …`
   - `KERNEL` is the Java method name of the task (`MyClass::reduce` → `reduce`).
   - `LINE` is a line of the *generated CUDA C*, not of the Java source. Take it from a
     sanitizer report, or run once with `-b KERNEL` (entry) and read `where.source`.
   - `--at 2:5` is block 2, thread 5 (1-D). `--at 2,0,0:5,1,0` is the 3-D form. `--at` reads threads
     *wherever they are* at the stop; a thread in a block that isn't resident yet reports "not resident".
   - To stop *on* a specific thread, use `-b KERNEL:LINE@B:T`. Use `-b "KERNEL:LINE if <cond>"` for any
     per-thread condition, e.g. `i_10 == 8`. Cost: cuda-gdb checks every warp that reaches the line, about 30 ms per
     block before the target. Deep into a large grid (thousands of blocks) that takes minutes, so prefer a smaller problem size.
   - Text output includes a **"locals that differ between the inspected threads"** table. Read it first.
   - The JSON has `batch.hits[].where` (function, line, code, source path), `batch.hits[].arrays`
     (array arguments with their element type), and `batch.hits[].threads[]` with `locals`, `hints` (the Java
     meaning of each local, e.g. `"f_8": "arg1[ctx.globalIdx]"`), `meaning` (of the current line), `shared` and `print`.
   - **Choosing threads:** for races, take threads of the *same block* in *different warps*
     (for example `--at 0:0 --at 0:64`) with `--hits 2` or more. If their loop variables differ at
     the same hit, one warp ran ahead, and there is a missing barrier. For out-of-bounds bugs, take the
     thread the sanitizer names, plus a known-good thread.
   - Only the thread that hit the breakpoint is guaranteed to be at `LINE`. The others are
     captured wherever they are, and their `line` shows where. A thread reported as "not resident/active" has
     usually finished already, which is itself evidence (for example a warp that ran ahead).
   - `__shared__` arrays show 16 elements by default. Use `--elements 256`, or
     `-p "((@shared float*)&'KERNEL::adf_2')[128]@8"`, to see more.
5. **Map back to Java.** Generated names are SSA-style (`i_3`, `f_8`, `ul_7`). Start from `hints`; it already
   recognises the patterns below. For anything else, read the generated source (`where.source`, or
   `~/.tornado-cuda-debug/sessions/<ts>/src/<kernel>.cu`):
   - `(blockIdx.x*blockDim.x+threadIdx.x)` → `ctx.globalIdx`; `threadIdx.x` → `ctx.localIdx`; `blockIdx.x` → `ctx.groupIdx`
   - `__shared__ float adf_N[SIZE]` → `ctx.allocateFloatLocalArray(SIZE)`; `__syncthreads()` → `ctx.localBarrier()`
   - `l = (long) i + 4; l << 2` then `ptr + l` → `array.get(i)` for a 4-byte element (the TornadoVM array header is 16 bytes)
   - `argN` is the N-th Java parameter counting from 0, `KernelContext` included. With `(KernelContext ctx, FloatArray a, …)`,
     `a` is `arg1`; in a `@Parallel` method `(FloatArray a, …)`, `a` is `arg0`.
6. **Fix the Java, then re-run the same sanitizer command** to confirm 0 errors, and run
   the program normally to check the result (`$TORNADOVM_HOME/bin/tornado -cp classes Main`).
   To compile against the SDK: `javac --release 21 --enable-preview -proc:none -cp "$TORNADOVM_HOME/share/java/tornado/*" -d classes src/*.java`.

## Other tools

- **Same kernel, two runs:** `tcd batch … --json > a.json`, then a second run with different inputs, then `tcd diff a.json b.json`.
  It compares by variable name, or by meaning when the two kernels differ (buggy vs fixed). It exits 1 if anything differs.
- **Long-running apps** (for example LLM inference): the user runs `tcd launch -- <java args>`. You then take snapshots
  with `tcd batch --pid PID -b KERNEL[:LINE] --json`, which attaches, reports and detaches, and the app keeps running.

## Interactive use (when the user drives)

- `tcd ui -b KERNEL[:LINE] -- …` starts a web debugger at http://127.0.0.1:7777. It has
  breakpoints in the gutter, F5/F10/F11, a block/thread focus picker, locals, shared memory,
  a TornadoVM array viewer and warps. On a remote machine, tunnel with `ssh -L 7777:127.0.0.1:7777 host`.
- VS Code: the extension in `vscode/` (`tcd dap`). `tcd attach PID` attaches to an app started with `tcd launch`.
- `tcd run -b KERNEL -- …` starts plain cuda-gdb. Useful commands there:
  `cuda block (b,0,0) thread (t,0,0)`, `next`, `tcd-list`, `tcd-array arg1 float 0 16`,
  `tcd-shared adf_2 float 0 32`, `info cuda warps`, `info cuda kernels`.

## Gotchas

- `-G` debug builds are slow and change scheduling. Under `-G` with no breakpoints, the racy example
  even produces *correct* results. Keep problem sizes small for breakpoint sessions, and trust racecheck
  (tcd builds with `-lineinfo` only) for races.
- A session starts in about 2 s, because tcd's agent skips TornadoVM's transfer warm-up. If a run seems to hang
  for about 20 s at start, check that `--no-fast-start` isn't set.
- `<unavailable>` means the variable is not live at that line in that thread. Variables that
  have not been assigned yet can also show *stale garbage* (a huge `l_23`, a `0` pointer)
  with no marker. Trust only variables assigned on or before the current `line`.
- A bare `break add` in raw cuda-gdb also hits JVM host functions named `add`. Always use
  `tcd-break` / `-b`.
- Batch mode stops at the first hit, and all resident warps stop with it. A thread that is not resident
  reports `"error": "thread not resident/active at this stop"`. Pick a thread in a block
  that the hit's `info cuda kernels` / focus shows as active, or use `--hits` to go further.
- tcd needs write access to `$TORNADOVM_HOME/var/tcd`, because TornadoVM always writes
  kernel dumps under `TORNADOVM_HOME`. Dumps are moved into the session directory afterwards.
- A timeout (`--timeout`, default 600 s) kills runaway batch sessions. If a run hits it,
  check `~/.tornado-cuda-debug/sessions/<ts>/gdb.log`.
