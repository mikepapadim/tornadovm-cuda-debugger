# tornado_gdb.py - cuda-gdb extension for debugging TornadoVM CUDA kernels.
#
# Loaded by tcd (tornado-cuda-debug). It can also be sourced by hand:
#   (cuda-gdb) source /path/to/tornado_gdb.py
#
# What it adds:
#   * Source mapping. TornadoVM compiles every kernel with NVRTC under the same
#     name, "tornado_kernel.cu", so debug info never points at a real file.
#     When a CUDA thread stops, the dumped source of the kernel it stopped in is
#     copied to <session>/src/<kernel>/tornado_kernel.cu and put on the source path.
#   * tcd-break KERNEL[:LINE]
#     A breakpoint that fires only in the named device kernel, never in host code
#     that happens to share the name (e.g. "add" also matches os::PageSizes::add).
#   * tcd-array / tcd-shared
#     Read TornadoVM array arguments (skipping their header) and __shared__ buffers.
#   * tcd-where / tcd-snapshot / tcd-batch
#     Machine-readable (JSON) views of the stop state, used by the CLI and web UI.
#
# Environment:
#   TCD_SESSION    session directory (src/ is created under it)
#   TCD_DUMP_ROOT  directory where TornadoVM writes <id>-<kernel>.cl dumps
#   TCD_BATCH      JSON spec for tcd-batch

import gdb
import glob
import json
import os
import re
import shutil

SESSION = os.environ.get("TCD_SESSION", os.path.join(os.getcwd(), ".tcd-session"))
DUMP_ROOT = os.environ.get("TCD_DUMP_ROOT", "")
JSON_MARK = "TCD-JSON:"
MAX_ELEMENTS = int(os.environ.get("TCD_MAX_ELEMENTS", "16"))

_FOCUS_RE = re.compile(r"kernel (\d+), (?:grid (-?\d+), )?block \((\d+),(\d+),(\d+)\), thread \((\d+),(\d+),(\d+)\)")
_current_source_dir = None


# ---------------------------------------------------------------- helpers

def _run(cmd):
    return gdb.execute(cmd, to_string=True)


def cuda_focus():
    """Returns {kernel, block:[x,y,z], thread:[x,y,z]} or None when focus is on the host."""
    try:
        out = _run("cuda kernel block thread")
    except gdb.error:
        return None
    m = _FOCUS_RE.search(out)
    if not m:
        return None
    g = [int(v) if v is not None else None for v in m.groups()]
    return {"kernel": g[0], "block": g[2:5], "thread": g[5:8]}


def frame_function():
    try:
        f = gdb.selected_frame()
        return f.name()
    except gdb.error:
        return None


def kernel_base_name(name):
    # cuda-gdb may render the kernel as "add" or "add<<<...>>>"
    return name.split("<<<")[0].split("(")[0].strip() if name else None


def find_dump(kernel):
    """Newest dumped source for a kernel. TornadoVM names dumps <task-id>-<entry>.cl."""
    if not DUMP_ROOT or not kernel:
        return None
    cands = glob.glob(os.path.join(DUMP_ROOT, "**", "*-" + kernel + ".cl"), recursive=True)
    cands += glob.glob(os.path.join(DUMP_ROOT, "**", kernel + ".cl"), recursive=True)
    if not cands:
        return None
    return max(cands, key=os.path.getmtime)


def map_source(kernel):
    """Makes tornado_kernel.cu resolve to this kernel's generated source. Returns the path or None."""
    global _current_source_dir
    dump = find_dump(kernel)
    if dump is None:
        return None
    target_dir = os.path.join(SESSION, "src", kernel)
    target = os.path.join(target_dir, "tornado_kernel.cu")
    os.makedirs(target_dir, exist_ok=True)
    if not os.path.exists(target) or os.path.getmtime(dump) > os.path.getmtime(target):
        shutil.copyfile(dump, target)
    if _current_source_dir != target_dir:
        # "directory DIR" also drops gdb's cached source text, so the next listing
        # of tornado_kernel.cu is read from the newly mapped kernel.
        _run("directory " + target_dir)
        _current_source_dir = target_dir
    return target


def current_location():
    try:
        sal = gdb.selected_frame().find_sal()
        return sal.line
    except gdb.error:
        return None


def source_line(path, line):
    if not path or not line:
        return None
    try:
        with open(path) as f:
            lines = f.read().splitlines()
        return lines[line - 1].strip() if 0 < line <= len(lines) else None
    except OSError:
        return None


def frame_locals():
    """Locals and arguments of the selected frame as {name: string}. -G keeps them all."""
    result = {}
    try:
        frame = gdb.selected_frame()
        block = frame.block()
    except (gdb.error, RuntimeError):
        return result
    while block is not None:
        for sym in block:
            if not (sym.is_variable or sym.is_argument) or sym.name in result:
                continue
            try:
                result[sym.name] = fmt(sym.value(frame))
            except (gdb.error, gdb.MemoryError) as e:
                result[sym.name] = "<%s>" % e
        if block.function is not None:
            break
        block = block.superblock
    return result


def fmt(value, max_elements=None):
    """Pointers as plain hex (TornadoVM passes arrays as unsigned char*, which gdb
would otherwise print as strings); arrays such as __shared__ buffers truncated."""
    max_elements = max_elements or MAX_ELEMENTS
    t = value.type.strip_typedefs()
    if t.code == gdb.TYPE_CODE_PTR:
        return hex(int(value))
    try:
        return value.format_string(max_elements=max_elements)
    except (AttributeError, TypeError):
        return str(value)


_SHARED_RE = re.compile(r"__shared__\s+([\w ]+?)\s+(\w+)\[(\d+)\]")


def shared_arrays(kernel, source_path, max_elements=None):
    """__shared__ arrays declared in the generated kernel, read in the focused block.
They have no debug type under NVRTC -G, so each is read through its symbol with an
explicit @shared cast: ((@shared T*)&'kernel::name')[0]@n."""
    max_elements = max_elements or MAX_ELEMENTS
    out = {}
    if not kernel or not source_path:
        return out
    try:
        with open(source_path) as f:
            decls = _SHARED_RE.findall(f.read())
    except OSError:
        return out
    for ctype, name, size in decls:
        n = min(int(size), max_elements)
        expr = "((@shared %s*)&'%s::%s')[0]@%d" % (ctype.strip(), kernel, name, n)
        try:
            text = gdb.parse_and_eval(expr).format_string(max_elements=0)
            out[name] = text if n == int(size) else text[:-1] + ", ...}"
        except gdb.error as e:
            out[name] = "<%s>" % e
    return out


# ---------------------------------------------------------------- generated-code hints
#
# TornadoVM's CUDA C is SSA-like: every Java expression becomes a chain of tiny assignments.
# These rules recover the Java meaning of the common chains, so a local reads
# "f_8 = 67  (arg1[ctx.globalIdx])" instead of just "f_8 = 67".

_BUILTINS = {
    "(blockIdx.x*blockDim.x+threadIdx.x)": "ctx.globalIdx",
    "(blockIdx.y*blockDim.y+threadIdx.y)": "ctx.globalIdy",
    "(blockIdx.z*blockDim.z+threadIdx.z)": "ctx.globalIdz",
    "(threadIdx.x)": "ctx.localIdx", "(threadIdx.y)": "ctx.localIdy", "(threadIdx.z)": "ctx.localIdz",
    "(blockIdx.x)": "ctx.groupIdx", "(blockIdx.y)": "ctx.groupIdy", "(blockIdx.z)": "ctx.groupIdz",
    "(blockDim.x)": "ctx.localGroupSizeX", "(blockDim.y)": "ctx.localGroupSizeY",
    "(gridDim.x*blockDim.x)": "ctx.globalGroupSizeX (grid stride)",
    "(gridDim.y*blockDim.y)": "ctx.globalGroupSizeY (grid stride)",
}
_ASSIGN_RE = re.compile(r"^\s*(\w+)\s*=\s*(.+?);\s*$")
_STORE_RE = re.compile(r"^\s*\*\(\(\s*([\w ]+?)\s*\*\)\s*(\w+)\)\s*=\s*(\w+);")
_LOAD_RE = re.compile(r"^\*\(\(\s*([\w ]+?)\s*\*\)\s*(\w+)\)$")
_CAST_RE = re.compile(r"^\((?:unsigned )?(?:long long|int|long|short|char|float|double)\)\s*(\w+)$")
_ADD_RE = re.compile(r"^(\w+)\s*\+\s*(\d+)L?$")
_SHL_RE = re.compile(r"^(\w+)\s*<<\s*(\d+)$")
_SUM_RE = re.compile(r"^(\w+)\s*\+\s*(\w+)$")
_SHARED_READ_RE = re.compile(r"^(adf|adi|adl|adb|adh|ads)_\d+\[(\w+)\]$")
_ARRAY_HEADER = int(os.environ.get("TCD_ARRAY_HEADER", "16"))


def analyze_source(path):
    """Returns (hints {var: java-ish meaning}, arrays {argN: element C type}, lines {line: meaning})."""
    try:
        with open(path) as f:
            text = f.read().splitlines()
    except (OSError, TypeError):
        return {}, {}, {}
    sym = {}      # var -> ("text", str) | ("arg", n) | ("add", v, k) | ("shl", v, k) | ("addr", n, off) | ("alias", v)
    hints, arrays, lines = {}, {}, {}

    def root(v):
        seen = 0
        while v in sym and sym[v][0] == "alias" and seen < 50:
            v, seen = sym[v][1], seen + 1
        return v

    def name(v):
        r = root(v)
        t = sym.get(r)
        return t[1] if t and t[0] == "text" else r

    def element(n, off):
        """off is the byte offset symbol inside argN; returns index text when it is (idx + H) << k."""
        o = sym.get(off)
        if o and o[0] == "alias":
            o = sym.get(root(off))
        if o and o[0] == "shl":
            inner = sym.get(o[1]) or sym.get(root(o[1]))
            if inner and inner[0] == "add" and (inner[2] << o[2]) == _ARRAY_HEADER:
                return name(inner[1])
        if o and o[0] == "add" and o[2] == _ARRAY_HEADER:
            return name(o[1])
        return None

    for no, line in enumerate(text, 1):
        st = _STORE_RE.match(line)
        if st:
            ctype, ptr, val = st.groups()
            p = sym.get(root(ptr))
            if p and p[0] == "addr":
                idx = element(p[1], p[2])
                arrays["arg%d" % p[1]] = ctype
                if idx:
                    lines[no] = "arg%d[%s] = %s" % (p[1], idx, name(val))
            continue
        m = _ASSIGN_RE.match(line)
        if not m:
            continue
        var, rhs = m.group(1), m.group(2).strip()
        if rhs in _BUILTINS:
            sym[var] = ("text", _BUILTINS[rhs])
            hints[var] = _BUILTINS[rhs]
            continue
        c = _CAST_RE.match(rhs)
        if c:
            src = c.group(1)
            if src.startswith("arg"):
                n = int(src[3:])
                sym[var] = ("arg", n)
                hints[var] = "base of arg%d" % n
            else:
                sym[var] = ("alias", src)
                if name(src) != root(src) or src in hints:
                    hints[var] = hints.get(root(src), name(src))
            continue
        a = _ADD_RE.match(rhs)
        if a:
            sym[var] = ("add", a.group(1), int(a.group(2)))
            continue
        sh = _SHL_RE.match(rhs)
        if sh:
            sym[var] = ("shl", sh.group(1), int(sh.group(2)))
            continue
        sm = _SUM_RE.match(rhs)
        if sm:
            b, off = sm.groups()
            if sym.get(root(b), ("",))[0] == "arg":
                sym[var] = ("addr", sym[root(b)][1], off)
                idx = element(sym[root(b)][1], off)
                if idx:
                    hints[var] = "&arg%d[%s]" % (sym[root(b)][1], idx)
            continue
        ld = _LOAD_RE.match(rhs)
        if ld:
            ctype, ptr = ld.groups()
            p = sym.get(root(ptr))
            if p and p[0] == "addr":
                arrays["arg%d" % p[1]] = ctype
                idx = element(p[1], p[2])
                if idx:
                    hints[var] = "arg%d[%s]" % (p[1], idx)
                    lines[no] = "%s = arg%d[%s]" % (var, p[1], idx)
            continue
        sr = _SHARED_READ_RE.match(rhs)
        if sr:
            hints[var] = "%s[%s] (__shared__)" % (rhs.split("[")[0], name(sr.group(2)))
    return hints, arrays, lines


def array_args(arrays, count=None):
    """First elements of every array argument whose element type is known from its loads/stores."""
    out = {}
    count = count or min(MAX_ELEMENTS, 16)
    for arg, ctype in sorted(arrays.items()):
        try:
            r = read_array(arg, ctype, 0, count)
            out["%s (%s[])" % (arg, ctype)] = "{" + ", ".join(r["values"]) + (", ...}" if len(r["values"]) == count else "}")
        except gdb.error as e:
            out[arg] = "<%s>" % e
    return out


def evaluate(exprs):
    out = {}
    for e in exprs:
        try:
            out[e] = fmt(gdb.parse_and_eval(e), max_elements=200)
        except gdb.error as err:
            out[e] = "<error: %s>" % err
    return out


def set_focus(block, thread):
    b = ",".join(str(v) for v in block)
    t = ",".join(str(v) for v in thread)
    try:
        _run("cuda block (%s) thread (%s)" % (b, t))
        return True
    except gdb.error:
        return False


def where():
    focus = cuda_focus()
    fn = kernel_base_name(frame_function())
    src = map_source(fn) if focus else None
    line = current_location()
    return {
        "device": focus is not None,
        "focus": focus,
        "function": fn,
        "line": line,
        "source": src,
        "code": source_line(src, line),
    }


def emit(obj):
    gdb.write(JSON_MARK + json.dumps(obj) + "\n")
    gdb.flush()


# ---------------------------------------------------------------- stop hook

def _on_stop(event):
    focus = cuda_focus()
    if focus is None:
        return
    map_source(kernel_base_name(frame_function()))


gdb.events.stop.connect(_on_stop)


def _on_breakpoint_modified(bp):
    # Fires on every hit (the hit count changes), so return quickly once the condition is in place.
    if isinstance(bp, KernelBreakpoint) and bp.wanted_condition and bp.condition != bp.wanted_condition:
        bp._apply_condition()


gdb.events.breakpoint_modified.connect(_on_breakpoint_modified)


# ---------------------------------------------------------------- breakpoints

class KernelBreakpoint(gdb.Breakpoint):
    """Stops only when a device thread of `kernel` reaches the location."""

    def __init__(self, kernel, line=None, condition=None):
        self.kernel = kernel
        self.kline = line
        # Both forms are scoped to the NVRTC file name, so they only resolve inside
        # device modules. A bare "add" would also match every host function called
        # add in the JVM's native libraries, and each of those hits would go through stop().
        spec = "tornado_kernel.cu:%d" % line if line else "tornado_kernel.cu:" + kernel
        super().__init__(spec, internal=False)
        # cuda-gdb evaluates a condition per GPU thread and focuses the thread that matches. On a
        # pending breakpoint the condition cannot be parsed yet (no device symbols), so it is
        # attached when the breakpoint resolves, which happens at module load, before any thread runs.
        self.wanted_condition = condition
        self._apply_condition()

    def _apply_condition(self):
        if self.wanted_condition and not self.pending and self.condition != self.wanted_condition:
            try:
                self.condition = self.wanted_condition
            except gdb.error as e:
                gdb.write("tcd: condition of breakpoint %d not usable yet: %s\n" % (self.number, e))

    def stop(self):
        # Runs on every warp that reaches the location, before gdb evaluates any condition,
        # so it has to stay cheap. The tornado_kernel.cu: location already restricts it to device
        # code, so the only question left is which kernel this is, since all kernels share that file name.
        try:
            name = gdb.selected_frame().name()
        except gdb.error:
            return False
        if kernel_base_name(name) != self.kernel:
            return False
        # Map the source now: stop() runs before gdb prints the stop location,
        # the stop event runs after it.
        map_source(self.kernel)
        return True


def _thread_condition(at):
    """'2:5' or '2,1,0:5,3,0' -> blockIdx/threadIdx condition.
Only the dimensions the user wrote are compared. cuda-gdb evaluates the condition on every warp that
reaches the line, and each extra builtin comparison costs a lot (6 terms are about 5x slower than 2).
Terms are ordered so the most selective one, the block, comes first."""
    b, t = at.split(":")
    bs = [v.strip() for v in b.split(",")][:3]
    ts = [v.strip() for v in t.split(",")][:3]
    parts = ["blockIdx.%s == %s" % (a, v) for a, v in zip("xyz", bs)] + ["threadIdx.%s == %s" % (a, v) for a, v in zip("xyz", ts)]
    return " && ".join(parts)


def parse_break_spec(spec):
    """KERNEL[:LINE][@BLOCK:THREAD][ if CONDITION] -> (kernel, line, condition)."""
    cond = None
    if " if " in spec:
        spec, cond = spec.split(" if ", 1)
        spec, cond = spec.strip(), cond.strip()
    at = None
    if "@" in spec:
        spec, at = spec.split("@", 1)
    kernel, line = (spec.rsplit(":", 1)[0], int(spec.rsplit(":", 1)[1])) if ":" in spec else (spec, None)
    if at:
        tc = _thread_condition(at)
        cond = "(%s) && (%s)" % (tc, cond) if cond else tc
    return kernel, line, cond


class TcdBreak(gdb.Command):
    """tcd-break KERNEL[:LINE][@B:T][ if COND] - break in a TornadoVM kernel (device code only).
LINE is a line of the generated CUDA C (see tcd-list). @2:5 stops only in block 2, thread 5
(3-D: @2,0,0:5,1,0). COND is any cuda-gdb condition, evaluated per GPU thread,
e.g. "tcd-break reduce:33 if i_10 == 8"."""

    def __init__(self):
        super().__init__("tcd-break", gdb.COMMAND_BREAKPOINTS)

    def invoke(self, arg, from_tty):
        if not arg.strip():
            raise gdb.GdbError("usage: tcd-break KERNEL[:LINE]")
        k, l, c = parse_break_spec(arg.strip())
        bp = KernelBreakpoint(k, l, c)
        gdb.write("tcd: breakpoint %d on kernel %s%s%s\n" % (bp.number, k, (" line %d" % l) if l else " (entry)", (" if " + c) if c else ""))


class TcdBreakpoints(gdb.Command):
    """tcd-breakpoints - JSON list of breakpoints with their kernel and generated-source line."""

    def __init__(self):
        super().__init__("tcd-breakpoints", gdb.COMMAND_BREAKPOINTS)

    def invoke(self, arg, from_tty):
        bps = []
        for bp in gdb.breakpoints():
            if not bp.visible:
                continue
            bps.append({
                "number": bp.number,
                "kernel": getattr(bp, "kernel", None),
                "line": getattr(bp, "kline", None),
                "condition": getattr(bp, "wanted_condition", None) or bp.condition,
                "location": bp.location,
                "hits": bp.hit_count,
                "enabled": bp.enabled,
            })
        emit({"breakpoints": bps})


class TcdWhere(gdb.Command):
    """tcd-where - JSON description of the current stop (focus, kernel, line, code)."""

    def __init__(self):
        super().__init__("tcd-where", gdb.COMMAND_STATUS)

    def invoke(self, arg, from_tty):
        emit({"where": where()})


class TcdSnapshot(gdb.Command):
    """tcd-snapshot [JSON] - locals of the focused thread, or of the threads listed in
JSON: {"at": [[bx,by,bz,tx,ty,tz], ...], "print": ["expr", ...]}"""

    def __init__(self):
        super().__init__("tcd-snapshot", gdb.COMMAND_DATA)

    def invoke(self, arg, from_tty):
        spec = json.loads(arg) if arg.strip() else {}
        emit({"snapshot": snapshot(spec.get("at"), spec.get("print", []))})


def snapshot(at, exprs):
    original = cuda_focus()
    threads = []
    targets = at or ([original["block"] + original["thread"]] if original else [])
    for coord in targets:
        entry = {"block": coord[0:3], "thread": coord[3:6]}
        if not set_focus(coord[0:3], coord[3:6]):
            entry["error"] = "thread not resident/active at this stop"
            threads.append(entry)
            continue
        w = where()
        hints, _, line_hints = analyze_source(w["source"])
        locs = frame_locals()
        entry.update({"line": w["line"], "code": w["code"], "meaning": line_hints.get(w["line"]), "locals": locs,
                      "hints": {k: v for k, v in hints.items() if k in locs},
                      "shared": shared_arrays(w["function"], w["source"]), "print": evaluate(exprs)})
        threads.append(entry)
    if original:
        set_focus(original["block"], original["thread"])
    return threads


class TcdBatch(gdb.Command):
    """tcd-batch - run the program non-interactively as described by $TCD_BATCH and
print one JSON document: every breakpoint hit with the requested thread state."""

    def __init__(self):
        super().__init__("tcd-batch", gdb.COMMAND_RUNNING)

    def invoke(self, arg, from_tty):
        spec = json.loads(os.environ.get("TCD_BATCH", "{}"))
        for b in spec.get("break", []):
            KernelBreakpoint(*parse_break_spec(b))
        max_hits = int(spec.get("hits", 1))
        report = {"hits": [], "kernels": None, "exit": None}
        attached = bool(spec.get("attach"))
        _run("continue" if attached else "run")
        while True:
            if gdb.selected_inferior().pid == 0:
                break
            focus = cuda_focus()
            if focus is None:
                # Stopped in host code (a real crash, not a JVM-internal signal).
                report["hostStop"] = _run("bt 8")
                break
            w = where()
            hit = {"where": w, "threads": snapshot(spec.get("at"), spec.get("print", []))}
            _, arrays, _ = analyze_source(w["source"])
            hit["arrays"] = array_args(arrays)
            if report["kernels"] is None:
                report["kernels"] = _run("info cuda kernels")
            report["hits"].append(hit)
            if len(report["hits"]) >= max_hits:
                break
            _run("continue")
        if gdb.selected_inferior().pid != 0:
            for b in [b for b in gdb.breakpoints() if isinstance(b, KernelBreakpoint)]:
                b.delete()
            try:
                _run("detach" if attached else "kill")
            except gdb.error:
                pass
            report["exit"] = ("detached" if attached else "killed") + " after %d hit(s)" % len(report["hits"])
        else:
            report["exit"] = "program exited"
        emit({"batch": report})


class TcdList(gdb.Command):
    """tcd-list - print the generated source of the focused kernel with line numbers."""

    def __init__(self):
        super().__init__("tcd-list", gdb.COMMAND_FILES)

    def invoke(self, arg, from_tty):
        w = where()
        if not w["source"]:
            raise gdb.GdbError("no mapped kernel source (is focus on a device thread?)")
        with open(w["source"]) as f:
            for i, text in enumerate(f.read().splitlines(), 1):
                mark = "=>" if i == w["line"] else "  "
                gdb.write("%s%4d  %s\n" % (mark, i, text))


ARRAY_HEADER = int(os.environ.get("TCD_ARRAY_HEADER", "16"))


def read_array(expr, ctype, start=0, count=16):
    """Elements of a TornadoVM native array (FloatArray, IntArray, ...) passed as a kernel arg.
The generated code sees these as raw byte pointers; the payload follows a header."""
    t = gdb.lookup_type(ctype)
    base = gdb.parse_and_eval("(unsigned long long)(%s)" % expr)
    addr = int(base) + ARRAY_HEADER + start * t.sizeof
    ptr = gdb.Value(addr).cast(t.pointer())
    values = []
    for i in range(count):
        try:
            values.append(str(ptr[i]))
        except gdb.MemoryError:
            values.append("<unreadable>")
            break
    return {"expr": expr, "type": ctype, "start": start, "values": values}


class TcdArray(gdb.Command):
    """tcd-array ARG TYPE [START [COUNT]] - show elements of a TornadoVM array argument.
Example: tcd-array arg1 float 0 8   (skips the 16-byte array header; override with TCD_ARRAY_HEADER)"""

    def __init__(self):
        super().__init__("tcd-array", gdb.COMMAND_DATA)

    def invoke(self, arg, from_tty):
        a = gdb.string_to_argv(arg)
        as_json = "--json" in a
        a = [x for x in a if x != "--json"]
        if len(a) < 2:
            raise gdb.GdbError("usage: tcd-array [--json] ARG TYPE [START [COUNT]]")
        r = read_array(a[0], a[1], int(a[2]) if len(a) > 2 else 0, int(a[3]) if len(a) > 3 else 16)
        if as_json:
            emit({"array": r})
        else:
            gdb.write("%s as %s[%d..]: %s\n" % (r["expr"], r["type"], r["start"], ", ".join(r["values"])))


class TcdShared(gdb.Command):
    """tcd-shared NAME TYPE [START [COUNT]] - show a __shared__ array of the focused kernel
in the focused block. Example: tcd-shared adf_2 float 120 16"""

    def __init__(self):
        super().__init__("tcd-shared", gdb.COMMAND_DATA)

    def invoke(self, arg, from_tty):
        a = gdb.string_to_argv(arg)
        if len(a) < 2:
            raise gdb.GdbError("usage: tcd-shared NAME TYPE [START [COUNT]]")
        kernel = kernel_base_name(frame_function())
        start = int(a[2]) if len(a) > 2 else 0
        count = int(a[3]) if len(a) > 3 else 16
        expr = "((@shared %s*)&'%s::%s')[%d]@%d" % (a[1], kernel, a[0], start, count)
        gdb.write("%s[%d..%d] = %s\n" % (a[0], start, start + count - 1, gdb.parse_and_eval(expr)))


for _cmd in (TcdBreak, TcdBreakpoints, TcdWhere, TcdSnapshot, TcdBatch, TcdList, TcdArray, TcdShared):
    _cmd()
