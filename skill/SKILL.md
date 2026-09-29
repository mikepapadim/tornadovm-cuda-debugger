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

   Each report comes with `>> kernel:LINE  <generated code>`. The summary at the end groups
   reports by line and prints a ready-made `tcd batch` command for the first failing thread.
4. **Inspect the failing thread.**
   `tcd batch -b KERNEL:LINE --at B:T [--at B:T] [--hits N] [-p expr,...] --json -- …`
   - `KERNEL` is the Java method name of the task (`MyClass::reduce` → `reduce`).
   - `LINE` is a line of the *generated CUDA C*, not of the Java source. Take it from a
     sanitizer report, or run once with `-b KERNEL` (entry) and read `where.source`.
   - `--at 2:5` is block 2, thread 5 (1-D). `--at 2,0,0:5,1,0` is the 3-D form.
   - The JSON has `batch.hits[].where` (function, line, code, source path) and
     `batch.hits[].threads[]` with `locals`, `shared` (`__shared__` arrays) and `print`.
   - Compare a correct thread with a failing one, or the same thread across `--hits 2+`.
     If threads of one block are at different loop iterations (for example the stride
     variable differs), there is a missing barrier.
5. **Map back to Java.** Generated names are SSA-style (`i_3`, `f_8`, `ul_7`). Read the
   generated source (the path in `where.source`, or `~/.tornado-cuda-debug/sessions/<ts>/dumps/`):
   - `(blockIdx.x*blockDim.x+threadIdx.x)` → `ctx.globalIdx`; `threadIdx.x` → `ctx.localIdx`; `blockIdx.x` → `ctx.groupIdx`
   - `__shared__ float adf_N[SIZE]` → `ctx.allocateFloatLocalArray(SIZE)`; `__syncthreads()` → `ctx.localBarrier()`
   - `l = (long) i + 4; l << 2` then `ptr + l` → `array.get(i)` for a 4-byte element (the TornadoVM array header is 16 bytes)
   - `argN` is the N-th non-`KernelContext` task parameter.
6. **Fix the Java, then re-run the same sanitizer command** to confirm 0 errors, and run
   the program normally to check the result.

## Interactive use (when the user drives)

- `tcd ui -b KERNEL[:LINE] -- …` starts a web debugger at http://127.0.0.1:7777. It has
  breakpoints in the gutter, F5/F10/F11, a block/thread focus picker, locals, shared memory,
  a TornadoVM array viewer and warps. On a remote machine, tunnel with `ssh -L 7777:127.0.0.1:7777 host`.
- `tcd run -b KERNEL -- …` starts plain cuda-gdb. Useful commands there:
  `cuda block (b,0,0) thread (t,0,0)`, `next`, `tcd-list`, `tcd-array arg1 float 0 16`,
  `tcd-shared adf_2 float 0 32`, `info cuda warps`, `info cuda kernels`.

## Gotchas

- `-G` debug builds are slow and change scheduling. Keep problem sizes small for
  breakpoint sessions, and trust racecheck (tcd builds with `-lineinfo` only) for races.
- `<unavailable>` means the variable is not live yet at that line in that thread. Step, or
  stop at a later line.
- A bare `break add` in raw cuda-gdb also hits JVM host functions named `add`. Always use
  `tcd-break` / `-b`.
- Batch mode stops at the first hit, and all resident warps stop with it. A thread that is not resident
  reports `"error": "thread not resident/active at this stop"`. Pick a thread in a block
  that the hit's `info cuda kernels` / focus shows as active, or use `--hits` to go further.
- tcd needs write access to `$TORNADOVM_HOME/var/tcd`, because TornadoVM always writes
  kernel dumps under `TORNADOVM_HOME`. Dumps are moved into the session directory afterwards.
- A timeout (`--timeout`, default 600 s) kills runaway batch sessions. If a run hits it,
  check `~/.tornado-cuda-debug/sessions/<ts>/gdb.log`.
