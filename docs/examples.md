# Worked examples

Build them first: `examples/build.sh` (needs `TORNADOVM_HOME`).

## Example 1: a missing barrier (`examples/src/SharedReduce.java`)

`reduceBuggy` sums 256 values per block in shared memory, but the author forgot
`ctx.localBarrier()`. Every block gets the wrong sum (3.0 instead of 256 on an RTX 4090):

```
$ tornado -cp examples/classes SharedReduce
partial[0] = 3.0 (expected 256)
reduceBuggy: 1024/1024 blocks wrong -> FAIL
```

**Step 1: find the bug.** Run racecheck. `tcd` maps each hazard to the generated CUDA C line:

```
$ bin/tcd memcheck --tool racecheck -- -cp examples/classes SharedReduce
========= Error: Race reported between Write access at reduceBuggy+0x160 in tornado_kernel.cu:20
=========         >> reduceBuggy:20  adf_2[i_9]  =  f_8;
=========     and Read access at reduceBuggy+0xf0 in tornado_kernel.cu:33 [524288 hazards]
=========         >> reduceBuggy:33  f_15  =  adf_2[i_14];
```

Line 20 is `scratch[lid] = input.get(gid)` and line 33 reads `scratch[lid + stride]`. There
is no `__syncthreads()` between the write and the read.

**Step 2: see it happen.** Stop at the read twice and compare two warps of block 0:

```
$ bin/tcd batch -b reduceBuggy:33 --hits 2 --at 0:0 --at 0:100 -- -cp examples/classes SharedReduce
== hit 2: reduceBuggy at line 33   f_15  =  adf_2[i_14];
-- block (0,0,0) thread (0,0,0)  (line 33)
   __shared__ adf_2     = {2 <repeats 16 times>, ...}
   i_10                 = 64          <- warp 0 is at stride 64
-- block (0,0,0) thread (100,0,0)  (line 28)
   __shared__ adf_2     = {2 <repeats 16 times>, ...}
   i_10                 = 8           <- warp 3 has already reached stride 8
```

The warps of one block are at different loop iterations, so the later strides read slots
that the other warps have not written yet. That is what a missing barrier looks like. The
**Warps** tab in `tcd ui` shows the same thing as warps at different PCs.

**Step 3: fix and confirm.** `reduceFixed` adds the two `ctx.localBarrier()` calls:

```
$ bin/tcd memcheck --tool racecheck -- -cp examples/classes SharedReduce fixed
reduceFixed: 0/1024 blocks wrong -> PASS
========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)
```

## Example 2: a silent out-of-bounds write (`examples/src/Saxpy.java`)

The grid is rounded up to a multiple of 256 (1,000,192 threads for 1,000,000 elements) and
`saxpyBuggy` has no bounds check. Run normally, the program prints `PASS`: the 192 tail threads
read and write past the end of `x` and `y`, and nothing complains. Under memcheck:

```
$ bin/tcd memcheck -- -cp examples/classes Saxpy
========= Invalid __global__ read of size 4 bytes
=========     at saxpyBuggy+0xb0 in tornado_kernel.cu:16
=========         >> saxpyBuggy:16  f_7  =  *(( float *) ul_6);
=========     by thread (160,0,0) in block (3906,0,0)
=========     Address 0x10009dd0aa0 is out of bounds
...
tcd memcheck summary (grouped by generated source line):
      32 x  Invalid __global__ read of size 4 bytes @ saxpyBuggy:16
             code:  f_7  =  *(( float *) ul_6);
             first: block (3906,0,0) thread (160,0,0)   -> tcd batch -b saxpyBuggy:16 --at 3906,0,0:160,0,0
```

Thread 160 of block 3906 has `globalIdx` = 3906·256 + 160 = 1,000,096, which is past `n`. (The order of
reports can differ between runs, so the first thread listed may be a different tail thread.) The
fix is `if (i < y.getSize())` (`saxpyFixed`), and memcheck then reports 0 errors.

## Limitations and tips

* Local names are the generated ones (`i_3`, `f_8`). `tcd-list`, or the source pane, shows
  which Java expression each one comes from. The `// BLOCK n` comments follow the Graal IR blocks.
* Values show as `<unavailable>` before their line has executed in that thread.
* If two task graphs compile the same Java method, they share a kernel name. `tcd` maps
  the most recently compiled one.
* `-G` makes kernels much slower. Keep breakpoint runs to small problem sizes, and use
  `tcd memcheck` (with `-lineinfo` only) to locate a bug first.
* `__shared__` arrays have no debug type under NVRTC, so `tcd` reads them through the
  symbol: `((@shared float*)&'kernel::name')[0]@n`. `tcd-shared` wraps this.
* A TornadoVM array argument is a raw `unsigned char*` to the object. The data starts
  after a 16-byte header, which `tcd-array` accounts for (override with `TCD_ARRAY_HEADER`).
* Tested with cuda-gdb 12.6, driver 610, RTX 4090 and TornadoVM 7.0.1 (JDK 21). It
  needs the TornadoVM **CUDA** backend. The older PTX backend loads PTX through the driver
  JIT without debug info.
