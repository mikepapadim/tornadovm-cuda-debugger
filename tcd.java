///usr/bin/env jbang "$0" "$@" ; exit $?
//JAVA 21+
//FILES gdb/tornado.gdbinit=gdb/tornado.gdbinit
//FILES gdb/tornado_gdb.py=gdb/tornado_gdb.py
//FILES agent/TcdAgent.java=agent/TcdAgent.java
//FILES web/index.html=web/index.html
//FILES web/app.js=web/app.js
//FILES web/style.css=web/style.css

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.PrintStream;
import java.io.Writer;
import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Stream;

/**
 * tcd - tornado-cuda-debug. Debug TornadoVM CUDA kernels with cuda-gdb.
 *
 * <pre>
 *   tcd doctor
 *   tcd run      [--break K[:L]]... -- -cp classes my.Main args
 *   tcd batch    --break K[:L] [--at B:T]... [--print e1,e2] [--hits N] [--json] -- ...
 *   tcd memcheck [--tool memcheck|racecheck|initcheck|synccheck] -- ...
 *   tcd ui       [--port 7777] [--break K[:L]]... -- ...
 * </pre>
 *
 * Runs as a single-file Java program: {@code jbang tcd.java}, or {@code java tcd.java} through bin/tcd.
 */
public class tcd {

    static final String VERSION = "0.1.0";
    static final PrintStream out = System.out;
    static final PrintStream err = System.err;

    // ------------------------------------------------------------------ options

    static final class Options {
        String command;
        String tornadoHome;
        List<String> breaks = new ArrayList<>();
        List<int[]> at = new ArrayList<>();
        List<String> print = new ArrayList<>();
        int hits = 1;
        int timeout = 600;
        int elements = 16;
        boolean fastStart = true;
        long pid;
        boolean json;
        boolean verbose;
        boolean tui;
        boolean noRun;
        boolean keep;
        boolean open = true;
        int port = 7777;
        List<String> tools = new ArrayList<>();
        String extraFlags = "";
        String cudaGdb;
        List<String> appArgs = new ArrayList<>();
    }

    static Options parse(String[] args) {
        Options o = new Options();
        int i = 0;
        if (args.length == 0) {
            usage(0);
        }
        o.command = args[i++];
        for (; i < args.length; i++) {
            String a = args[i];
            switch (a) {
                case "--" -> {
                    o.appArgs.addAll(List.of(args).subList(i + 1, args.length));
                    i = args.length;
                }
                case "--tornado-home" -> o.tornadoHome = args[++i];
                case "--break", "-b" -> o.breaks.add(args[++i]);
                case "--at" -> o.at.add(parseCoord(args[++i]));
                case "--print", "-p" -> o.print.addAll(List.of(args[++i].split(",")));
                case "--hits" -> o.hits = Integer.parseInt(args[++i]);
                case "--timeout" -> o.timeout = Integer.parseInt(args[++i]);
                case "--elements" -> o.elements = Integer.parseInt(args[++i]);
                case "--no-fast-start" -> o.fastStart = false;
                case "--pid" -> o.pid = Long.parseLong(args[++i]);
                case "--json" -> o.json = true;
                case "--verbose", "-v" -> o.verbose = true;
                case "--tui" -> o.tui = true;
                case "--no-run" -> o.noRun = true;
                case "--keep" -> o.keep = true;
                case "--no-open" -> o.open = false;
                case "--port" -> o.port = Integer.parseInt(args[++i]);
                case "--tool" -> o.tools.add(args[++i]);
                case "--flags" -> o.extraFlags = args[++i];
                case "--cuda-gdb" -> o.cudaGdb = args[++i];
                case "-h", "--help" -> usage(0);
                default -> {
                    // No "--": everything from the first unknown token is the application.
                    o.appArgs.addAll(List.of(args).subList(i, args.length));
                    i = args.length;
                }
            }
        }
        return o;
    }

    /** "2:5" (1-D), "2,0,0:5,0,0" or "b(2,0,0)t(5,0,0)" -> {bx,by,bz,tx,ty,tz}. */
    static int[] parseCoord(String s) {
        String[] parts = s.replaceAll("[a-zA-Z()\\s]", " ").trim().split("[:\\s]+");
        if (parts.length != 2) {
            throw new IllegalArgumentException("--at expects BLOCK:THREAD, e.g. 2:5 or 2,0,0:5,0,0 (got " + s + ")");
        }
        int[] r = new int[6];
        String[] b = parts[0].split(","), t = parts[1].split(",");
        for (int k = 0; k < 3; k++) {
            r[k] = k < b.length ? Integer.parseInt(b[k].trim()) : 0;
            r[3 + k] = k < t.length ? Integer.parseInt(t[k].trim()) : 0;
        }
        return r;
    }

    static void usage(int code) {
        (code == 0 ? out : err).println("""
                tcd %s - debug TornadoVM CUDA kernels with cuda-gdb

                Usage: tcd <command> [options] -- <java application args>

                Commands:
                  doctor     check cuda-gdb, driver, TornadoVM and permissions
                  run        interactive cuda-gdb session with TornadoVM pre-wired
                  batch      run to a kernel breakpoint and report thread state (text or --json)
                  memcheck   run under compute-sanitizer; reports mapped to generated source lines
                  ui         web debugger on http://127.0.0.1:PORT
                  launch     start a long-running app with debug kernels, ready for `attach` (no debugger yet)
                  attach PID interactive cuda-gdb on an app started with `tcd launch` (detach leaves it running)
                  dap        Debug Adapter Protocol server on stdin/stdout (VS Code, IntelliJ + LSP4IJ, ...)
                  diff A B   compare two `batch --json` reports (e.g. buggy vs fixed), thread by thread
                  version    print version

                Options:
                  --break, -b SPEC    break in device kernel K (a Java method name). SPEC is
                                      K[:L][@B:T][ if COND]: line L of the generated CUDA C, only
                                      block B / thread T, or any per-thread condition. Repeatable.
                                      e.g. -b reduce:33@0:64   -b "reduce:33 if i_10 == 8"
                  --at B:T            thread to inspect: 1-D "2:5" or 3-D "2,0,0:5,0,0". Repeatable.
                  --print, -p e1,e2   extra expressions to evaluate per thread
                  --hits N            batch: stop after N breakpoint hits (default 1)
                  --json              batch/memcheck: machine-readable output (memcheck: summary only,
                                      full reports in the session's sanitizer-<tool>.log)
                  --pid PID           batch: attach to an app started with `tcd launch` instead of starting one
                  --timeout SECS      batch: kill the session after SECS seconds (default 600)
                  --no-fast-start     do not skip TornadoVM's CUDA transfer warm-up (see agent/TcdAgent.java)
                  --elements N        batch/run/ui: array elements shown for locals and __shared__ (default 16)
                  --tool NAME         memcheck: sanitizer tool(s) (memcheck, racecheck, initcheck, synccheck)
                  --verbose, -v       memcheck: print every sanitizer report, not only the first 3
                                      memcheck exits 1 on sanitizer errors, else the app's exit code (CI)
                  --flags "..."       extra NVRTC flags (added to -G -lineinfo)
                  --tui               run: start cuda-gdb in TUI mode
                  --no-run            run/ui: do not start the program automatically
                  --port N            ui: HTTP port (default 7777, bound to 127.0.0.1)
                  --no-open           ui: do not try to open a browser
                  --keep              keep the session directory's dumps in $TORNADOVM_HOME too
                  --tornado-home DIR  TornadoVM SDK (default: $TORNADOVM_HOME, then `tornado` on PATH)
                  --cuda-gdb PATH     cuda-gdb binary (default: on PATH or /usr/local/cuda/bin)

                Examples:
                  tcd batch -b add --at 2:5 -- -cp classes VAdd
                  tcd run -b reduce:23 -- -cp app.jar my.Main
                  tcd memcheck --tool racecheck -- -cp classes Reduce
                  tcd ui -- -cp classes VAdd
                """.formatted(VERSION));
        System.exit(code);
    }

    // ------------------------------------------------------------------ main

    public static void main(String[] args) throws Exception {
        Options o = parse(args);
        int rc = switch (o.command) {
            case "doctor" -> doctor(o);
            case "run" -> run(o);
            case "batch" -> batch(o);
            case "memcheck" -> memcheck(o);
            case "diff" -> diff(o);
            case "dap" -> Dap.serve(o);
            case "launch" -> launch(o);
            case "attach" -> attach(o);
            case "ui" -> ui(o);
            case "version", "--version" -> {
                out.println("tcd " + VERSION);
                yield 0;
            }
            case "help", "-h", "--help" -> {
                usage(0);
                yield 0;
            }
            default -> {
                err.println("tcd: unknown command '" + o.command + "'");
                usage(2);
                yield 2;
            }
        };
        System.exit(rc);
    }

    // ------------------------------------------------------------------ environment

    static Path home() {
        String h = System.getProperty("tcd.home", System.getenv("TCD_HOME"));
        if (h != null) {
            return Path.of(h);
        }
        // java tcd.java / jbang from the repo directory
        for (Path p = Path.of("").toAbsolutePath(); p != null; p = p.getParent()) {
            if (Files.exists(p.resolve("gdb/tornado_gdb.py")) && Files.exists(p.resolve("tcd.java"))) {
                return p;
            }
        }
        return null;
    }

    /** A bundled file: from the repo when running from source, else the JBang classpath copy. */
    static Path resource(String name) throws IOException {
        Path h = home();
        if (h != null && Files.exists(h.resolve(name))) {
            return h.resolve(name);
        }
        try (InputStream in = tcd.class.getClassLoader().getResourceAsStream(name)) {
            if (in == null) {
                throw new IOException("cannot locate bundled file " + name + " (set TCD_HOME to the tcd checkout)");
            }
            Path tmp = Path.of(System.getProperty("java.io.tmpdir"), "tcd-" + VERSION, name);
            Files.createDirectories(tmp.getParent());
            Files.copy(in, tmp, StandardCopyOption.REPLACE_EXISTING);
            return tmp;
        }
    }

    static Optional<Path> which(String exe) {
        for (String dir : System.getenv().getOrDefault("PATH", "").split(":")) {
            Path p = Path.of(dir, exe);
            if (Files.isExecutable(p)) {
                return Optional.of(p);
            }
        }
        return Optional.empty();
    }

    static Optional<Path> cudaTool(String exe, String override) {
        if (override != null) {
            return Optional.of(Path.of(override));
        }
        Optional<Path> p = which(exe);
        if (p.isPresent()) {
            return p;
        }
        for (String c : new String[] { System.getenv("CUDA_HOME"), System.getenv("CUDA_PATH"), "/usr/local/cuda" }) {
            if (c != null && Files.isExecutable(Path.of(c, "bin", exe))) {
                return Optional.of(Path.of(c, "bin", exe));
            }
        }
        return Optional.empty();
    }

    static Path tornadoHome(Options o) {
        String h = o.tornadoHome != null ? o.tornadoHome : System.getenv("TORNADOVM_HOME");
        if (h == null) {
            h = which("tornado").map(p -> p.toAbsolutePath().getParent().getParent().toString()).orElse(null);
        }
        if (h == null || !Files.isExecutable(Path.of(h, "bin", "tornado"))) {
            throw new IllegalStateException("TornadoVM not found: set TORNADOVM_HOME or pass --tornado-home (see `tcd doctor`)");
        }
        return Path.of(h).toAbsolutePath().normalize();
    }

    static String exec(List<String> cmd, Map<String, String> env) {
        try {
            ProcessBuilder pb = new ProcessBuilder(cmd).redirectErrorStream(true);
            pb.environment().putAll(env);
            Process p = pb.start();
            String s = new String(p.getInputStream().readAllBytes(), StandardCharsets.UTF_8);
            p.waitFor(60, TimeUnit.SECONDS);
            return s;
        } catch (IOException | InterruptedException e) {
            return "";
        }
    }

    /** The JVM command line TornadoVM would use, without application arguments. */
    static List<String> tornadoJava(Path th) {
        String s = exec(List.of(th.resolve("bin/tornado").toString(), "--printJavaFlags"), Map.of("TORNADOVM_HOME", th.toString()));
        String line = s.lines().filter(l -> l.contains("bin/java ") || l.startsWith("java ")).reduce((a, b) -> b).orElse(null);
        if (line == null) {
            throw new IllegalStateException("`tornado --printJavaFlags` gave no java command:\n" + s);
        }
        return new ArrayList<>(List.of(line.trim().split("\\s+")));
    }

    // ------------------------------------------------------------------ session

    static final class Session {
        final Path dir;          // ~/.tornado-cuda-debug/sessions/<ts>
        final Path tornadoHome;
        final String dumpRel;    // relative to TORNADOVM_HOME (TornadoVM always prefixes it)
        final Path dumpAbs;
        final boolean keep;
        int elements = 16;
        boolean attachable;

        Session(Path th, boolean keep) throws IOException {
            String ts = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd-HHmmss-SSS"));
            this.dir = Path.of(System.getProperty("user.home"), ".tornado-cuda-debug", "sessions", ts);
            this.tornadoHome = th;
            this.dumpRel = "var/tcd/" + ts;
            this.dumpAbs = th.resolve(dumpRel);
            this.keep = keep;
            Files.createDirectories(dir);
        }

        Map<String, String> env() {
            Map<String, String> e = new LinkedHashMap<>();
            e.put("TORNADOVM_HOME", tornadoHome.toString());
            e.put("TCD_SESSION", dir.toString());
            e.put("TCD_DUMP_ROOT", dumpAbs.toString());
            e.put("TCD_MAX_ELEMENTS", String.valueOf(elements));
            return e;
        }

        /** The application's JVM command with debug compilation and source dumping switched on. */
        List<String> javaCommand(Options o, String nvrtcFlags) {
            List<String> cmd = tornadoJava(tornadoHome);
            List<String> extra = new ArrayList<>(List.of(
                    "-Dtornado.cuda.compiler.flags=" + (nvrtcFlags + " " + o.extraFlags).trim(),
                    "-Dtornado.opencl.source.dump=true",
                    "-Dtornado.opencl.source.dir=" + dumpRel,
                    // A cubin cached from a non-debug run must never be picked up.
                    "-Dtornado.cuda.codecache.enable=false"));
            if (o.fastStart || attachable) {
                Path agent = startupAgent(tornadoHome);
                if (agent != null) {
                    extra.add("-javaagent:" + agent + "=" + (o.fastStart ? "fast" : "") + (attachable ? ",ptracer" : ""));
                } else if (attachable) {
                    err.println("tcd: could not build the agent; attaching needs ptrace_scope 0 or root");
                }
            }
            if (attachable) {
                // The agent's prctl downcall comes from the unnamed module; TornadoVM grants native
                // access to tornado.runtime only.
                boolean found = false;
                for (int i = 0; i < cmd.size(); i++) {
                    if (cmd.get(i).startsWith("--enable-native-access=")) {
                        cmd.set(i, cmd.get(i) + ",ALL-UNNAMED");
                        found = true;
                    }
                }
                if (!found) {
                    extra.add("--enable-native-access=ALL-UNNAMED");
                }
            }
            cmd.addAll(1, extra);
            cmd.addAll(o.appArgs);
            return cmd;
        }

        /** Move the dumped kernels into the session so $TORNADOVM_HOME stays clean. */
        void close() {
            if (!Files.isDirectory(dumpAbs)) {
                return;
            }
            try (Stream<Path> files = Files.walk(dumpAbs)) {
                Path target = dir.resolve("dumps");
                for (Path p : files.filter(Files::isRegularFile).toList()) {
                    Path t = target.resolve(dumpAbs.relativize(p).toString());
                    Files.createDirectories(t.getParent());
                    Files.copy(p, t, StandardCopyOption.REPLACE_EXISTING);
                }
                // Same sources under the name a developer looks for: src/<kernel>.cu
                Path src = dir.resolve("src");
                for (String k : dumpedKernels(target)) {
                    Path f = findDump(target, k);
                    if (f != null) {
                        Files.createDirectories(src);
                        Files.copy(f, src.resolve(k + ".cu"), StandardCopyOption.REPLACE_EXISTING);
                    }
                }
                if (!keep) {
                    deleteTree(dumpAbs);
                }
            } catch (IOException e) {
                err.println("tcd: could not collect kernel dumps: " + e.getMessage());
            }
        }
    }

    /**
     * Builds (once, cached) the javaagent that removes TornadoVM's CUDA transfer warm-up: about
     * 5,000 driver calls that take microseconds natively but about 20 s under cuda-gdb. It is compiled
     * here against the SDK's own ASM jar, so no binary is shipped. Returns null (and the session
     * just starts slower) if anything is missing.
     */
    static Path startupAgent(Path th) {
        try (Stream<Path> jars = Files.list(th.resolve("share/java/tornado"))) {
            Path asm = jars.filter(p -> p.getFileName().toString().matches("asm-\\d[\\d.]*\\.jar")).findFirst().orElse(null);
            javax.tools.JavaCompiler javac = javax.tools.ToolProvider.getSystemJavaCompiler();
            if (asm == null || javac == null) {
                return null;
            }
            Path src = resource("agent/TcdAgent.java");
            String key = Integer.toHexString((Files.readString(src) + asm.getFileName()).hashCode());
            Path dir = Path.of(System.getProperty("user.home"), ".tornado-cuda-debug", "agent");
            Path jar = dir.resolve("tcd-agent-" + key + ".jar");
            if (Files.exists(jar)) {
                return jar;
            }
            Path classes = Files.createTempDirectory("tcd-agent");
            int rc = javac.run(null, null, err, "--release", "21", "-nowarn", "-cp", asm.toString(), "-d", classes.toString(), src.toString());
            if (rc != 0) {
                return null;
            }
            Files.createDirectories(dir);
            java.util.jar.Manifest mf = new java.util.jar.Manifest();
            mf.getMainAttributes().put(java.util.jar.Attributes.Name.MANIFEST_VERSION, "1.0");
            mf.getMainAttributes().putValue("Premain-Class", "TcdAgent");
            Path tmp = dir.resolve(jar.getFileName() + ".tmp");
            try (java.util.jar.JarOutputStream jos = new java.util.jar.JarOutputStream(Files.newOutputStream(tmp), mf); Stream<Path> cs = Files.walk(classes)) {
                for (Path c : cs.filter(Files::isRegularFile).toList()) {
                    jos.putNextEntry(new java.util.jar.JarEntry(classes.relativize(c).toString().replace('\\', '/')));
                    jos.write(Files.readAllBytes(c));
                    jos.closeEntry();
                }
            }
            Files.move(tmp, jar, StandardCopyOption.REPLACE_EXISTING);
            deleteTree(classes);
            return jar;
        } catch (IOException | RuntimeException e) {
            return null;
        }
    }

    static void deleteTree(Path p) throws IOException {
        try (Stream<Path> s = Files.walk(p)) {
            for (Path q : s.sorted(Comparator.reverseOrder()).toList()) {
                Files.deleteIfExists(q);
            }
        }
    }

    static List<String> gdbBase(Options o, boolean mi) throws IOException {
        Path gdb = cudaTool("cuda-gdb", o.cudaGdb).orElseThrow(() -> new IllegalStateException("cuda-gdb not found (see `tcd doctor`)"));
        List<String> cmd = new ArrayList<>(List.of(gdb.toString(), "-q"));
        if (mi) {
            cmd.add("--interpreter=mi3");
        }
        cmd.addAll(List.of("-x", resource("gdb/tornado.gdbinit").toString(), "-x", resource("gdb/tornado_gdb.py").toString()));
        return cmd;
    }

    static void requireApp(Options o) {
        if (o.appArgs.isEmpty()) {
            err.println("tcd: no application given. Pass the java arguments after --, e.g.  tcd " + o.command + " -- -cp classes my.Main");
            System.exit(2);
        }
    }

    // ------------------------------------------------------------------ doctor

    static int doctor(Options o) {
        int problems = 0;
        out.println("tcd " + VERSION + " doctor");
        Optional<Path> gdb = cudaTool("cuda-gdb", o.cudaGdb);
        problems += check(gdb.isPresent(), "cuda-gdb", gdb.map(p -> p + "  " + firstLine(exec(List.of(p.toString(), "--version"), Map.of()), "NVIDIA")).orElse("not found"),
                "install the CUDA toolkit and put its bin/ on PATH (or pass --cuda-gdb)");
        Optional<Path> san = cudaTool("compute-sanitizer", null);
        check(san.isPresent(), "compute-sanitizer", san.map(Path::toString).orElse("not found (tcd memcheck unavailable)"), null);
        Optional<Path> smi = which("nvidia-smi");
        String gpu = smi.map(p -> exec(List.of(p.toString(), "--query-gpu=name,driver_version", "--format=csv,noheader"), Map.of()).trim()).orElse("");
        problems += check(!gpu.isEmpty() && !gpu.contains("failed"), "GPU / driver", gpu.isEmpty() ? "nvidia-smi not found" : gpu.lines().findFirst().orElse(gpu),
                "an NVIDIA GPU with a working driver is required");
        try {
            Path th = tornadoHome(o);
            problems += check(true, "TornadoVM", th.toString(), null);
            boolean cuda = Files.exists(th.resolve("etc/exportLists/cuda-exports")) || listContains(th.resolve("share/java/tornado"), "tornado-drivers-cuda");
            problems += check(cuda, "CUDA backend", cuda ? "present" : "missing", "use a TornadoVM SDK built with the CUDA backend (e.g. sdk install tornadovm 7.0.1-jdk21-cuda)");
            boolean writable = Files.isWritable(th) || Files.isWritable(th.resolve("var"));
            problems += check(writable, "kernel dump dir", th.resolve("var/tcd") + (writable ? " (writable)" : " NOT writable"),
                    "TornadoVM writes kernel sources under $TORNADOVM_HOME; use an SDK copy you own");
            List<String> java = tornadoJava(th);
            problems += check(true, "JVM", java.get(0), null);
        } catch (RuntimeException e) {
            problems += check(false, "TornadoVM", e.getMessage(), "set TORNADOVM_HOME to a TornadoVM SDK");
        }
        String ptrace = readQuiet(Path.of("/proc/sys/kernel/yama/ptrace_scope")).trim();
        check(!ptrace.equals("3"), "ptrace_scope", ptrace.isEmpty() ? "n/a" : ptrace + (ptrace.equals("3") ? " (attach disabled)" : " (launching under cuda-gdb is fine)"),
                "ptrace is fully disabled; set /proc/sys/kernel/yama/ptrace_scope to 1");
        String gdbLock = readQuiet(Path.of("/proc/driver/nvidia/params"));
        if (gdbLock.contains("NVreg_RestrictProfilingToAdminUsers: 1")) {
            check(true, "profiling", "restricted to admin (fine for cuda-gdb; may affect compute-sanitizer)", null);
        }
        try {
            problems += check(true, "tcd files", resource("gdb/tornado_gdb.py").getParent().getParent().toString(), null);
        } catch (IOException e) {
            problems += check(false, "tcd files", e.getMessage(), "run from the repo or set TCD_HOME");
        }
        out.println(problems == 0 ? "\nAll good. Try:  tcd batch -b <kernelMethod> -- -cp <classes> <Main>" : "\n" + problems + " problem(s) found.");
        return problems == 0 ? 0 : 1;
    }

    static int check(boolean ok, String what, String detail, String fix) {
        out.printf("  %s %-18s %s%n", ok ? "[ok]  " : "[FAIL]", what, detail);
        if (!ok && fix != null) {
            out.println("         -> " + fix);
        }
        return ok ? 0 : 1;
    }

    static String firstLine(String s, String containing) {
        return s.lines().filter(l -> l.contains(containing)).findFirst().orElse(s.lines().findFirst().orElse("")).trim();
    }

    static boolean listContains(Path dir, String prefix) {
        try (Stream<Path> s = Files.list(dir)) {
            return s.anyMatch(p -> p.getFileName().toString().startsWith(prefix));
        } catch (IOException e) {
            return false;
        }
    }

    static String readQuiet(Path p) {
        try {
            return Files.readString(p);
        } catch (IOException e) {
            return "";
        }
    }

    // ------------------------------------------------------------------ run (interactive)

    static int run(Options o) throws Exception {
        requireApp(o);
        Session s = new Session(tornadoHome(o), o.keep);
        s.elements = o.elements;
        List<String> cmd = gdbBase(o, false);
        if (o.tui) {
            cmd.add("-tui");
        }
        for (String b : o.breaks) {
            cmd.addAll(List.of("-ex", "tcd-break " + b));
        }
        if (!o.noRun) {
            cmd.addAll(List.of("-ex", "run"));
        }
        cmd.add("--args");
        cmd.addAll(s.javaCommand(o, "-G -lineinfo"));
        err.println("tcd: session " + s.dir);
        err.println("tcd: extra commands: tcd-break K[:L], tcd-list, tcd-where, tcd-snapshot, tcd-array ARG TYPE [START [N]]");
        err.println("tcd: CUDA focus:     cuda block (x,y,z) thread (x,y,z) | info cuda kernels | info cuda warps");
        ProcessBuilder pb = new ProcessBuilder(cmd).inheritIO();
        pb.environment().putAll(s.env());
        int rc = pb.start().waitFor();
        s.close();
        return rc;
    }

    // ------------------------------------------------------------------ launch / attach

    static Path launchedFile(long pid) {
        return Path.of(System.getProperty("user.home"), ".tornado-cuda-debug", "launched", pid + ".json");
    }

    /** Runs the app in the foreground with debug kernels and attach permission; prints how to attach. */
    static int launch(Options o) throws Exception {
        requireApp(o);
        Session s = new Session(tornadoHome(o), o.keep);
        s.elements = o.elements;
        s.attachable = true;
        ProcessBuilder pb = new ProcessBuilder(s.javaCommand(o, "-G -lineinfo")).inheritIO();
        pb.environment().putAll(s.env());
        Process p = pb.start();
        Path info = launchedFile(p.pid());
        Files.createDirectories(info.getParent());
        Files.writeString(info, Json.write(Map.of("session", s.dir.toString(), "dumpRoot", s.dumpAbs.toString(), "tornadoHome", s.tornadoHome.toString())));
        err.println("tcd: launched pid " + p.pid() + " with debug kernels (session " + s.dir + ")");
        err.println("tcd: in another terminal:  tcd attach " + p.pid() + " -b <kernel>[:line]     or  tcd batch --pid " + p.pid() + " -b <kernel> --json");
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            p.destroy();
            s.close();
            try {
                Files.deleteIfExists(info);
            } catch (IOException ignored) {
            }
        }));
        return p.waitFor();
    }

    /** Environment of a session started by `tcd launch` for this pid. */
    static Map<String, String> launchedEnv(long pid) throws IOException {
        Path info = launchedFile(pid);
        if (!Files.exists(info)) {
            throw new IllegalStateException("pid " + pid + " was not started with `tcd launch`, so its kernels have no debug info and it may refuse ptrace");
        }
        Object doc = Json.parse(Files.readString(info));
        Map<String, String> env = new LinkedHashMap<>();
        env.put("TORNADOVM_HOME", String.valueOf(Json.path(doc, "tornadoHome")));
        env.put("TCD_SESSION", String.valueOf(Json.path(doc, "session")));
        env.put("TCD_DUMP_ROOT", String.valueOf(Json.path(doc, "dumpRoot")));
        return env;
    }

    static int attach(Options o) throws Exception {
        if (o.pid == 0 && !o.appArgs.isEmpty()) {
            o.pid = Long.parseLong(o.appArgs.get(0));
        }
        if (o.pid == 0) {
            err.println("usage: tcd attach PID [-b kernel[:line]]...   (PID from `tcd launch`)");
            return 2;
        }
        Map<String, String> env = launchedEnv(o.pid);
        env.put("TCD_MAX_ELEMENTS", String.valueOf(o.elements));
        List<String> cmd = gdbBase(o, false);
        if (o.tui) {
            cmd.add("-tui");
        }
        for (String b : o.breaks) {
            cmd.addAll(List.of("-ex", "tcd-break " + b));
        }
        if (!o.noRun && !o.breaks.isEmpty()) {
            cmd.addAll(List.of("-ex", "continue"));
        }
        cmd.addAll(List.of("-p", String.valueOf(o.pid)));
        err.println("tcd: attaching to " + o.pid + ". `detach` leaves the app running, `quit` detaches too.");
        ProcessBuilder pb = new ProcessBuilder(cmd).inheritIO();
        pb.environment().putAll(env);
        return pb.start().waitFor();
    }

    // ------------------------------------------------------------------ batch

    static int batch(Options o) throws Exception {
        if (o.pid == 0) {
            requireApp(o);
        }
        if (o.breaks.isEmpty()) {
            err.println("tcd batch: give at least one --break KERNEL[:LINE]");
            return 2;
        }
        Session s = o.pid != 0 ? null : new Session(tornadoHome(o), o.keep);
        List<String> cmd = gdbBase(o, false);
        Map<String, String> env;
        if (s == null) {
            env = launchedEnv(o.pid);
            cmd.addAll(List.of("-batch", "-ex", "tcd-batch", "-p", String.valueOf(o.pid)));
        } else {
            s.elements = o.elements;
            env = s.env();
            cmd.addAll(List.of("-batch", "-ex", "tcd-batch", "--args"));
            cmd.addAll(s.javaCommand(o, "-G -lineinfo"));
        }
        env.put("TCD_MAX_ELEMENTS", String.valueOf(o.elements));
        Path sessionDir = Path.of(env.get("TCD_SESSION"));

        StringBuilder spec = new StringBuilder("{\"break\":[");
        for (int i = 0; i < o.breaks.size(); i++) {
            spec.append(i > 0 ? "," : "").append(Json.quote(o.breaks.get(i)));
        }
        spec.append("],\"hits\":").append(o.hits).append(",\"print\":[");
        for (int i = 0; i < o.print.size(); i++) {
            spec.append(i > 0 ? "," : "").append(Json.quote(o.print.get(i).trim()));
        }
        spec.append("]");
        if (!o.at.isEmpty()) {
            spec.append(",\"at\":[");
            for (int i = 0; i < o.at.size(); i++) {
                int[] c = o.at.get(i);
                spec.append(i > 0 ? "," : "").append("[").append(c[0]).append(',').append(c[1]).append(',').append(c[2]).append(',')
                        .append(c[3]).append(',').append(c[4]).append(',').append(c[5]).append("]");
            }
            spec.append("]");
        }
        if (o.pid != 0) {
            spec.append(",\"attach\":true");
        }
        spec.append("}");

        ProcessBuilder pb = new ProcessBuilder(cmd).redirectErrorStream(true);
        pb.environment().putAll(env);
        pb.environment().put("TCD_BATCH", spec.toString());
        if (!o.json) {
            err.println("tcd: " + (s == null ? "attached to " + o.pid : "running under cuda-gdb") + " (session " + sessionDir + ") ...");
        }
        Process p = pb.start();
        Thread watchdog = Thread.ofVirtual().start(() -> {
            try {
                if (!p.waitFor(o.timeout, TimeUnit.SECONDS)) {
                    err.println("tcd: timeout after " + o.timeout + "s, killing cuda-gdb (raise with --timeout)");
                    p.descendants().forEach(ProcessHandle::destroyForcibly);
                    p.destroyForcibly();
                }
            } catch (InterruptedException ignored) {
            }
        });
        String report = null;
        StringBuilder log = new StringBuilder();
        try (BufferedReader r = new BufferedReader(new InputStreamReader(p.getInputStream(), StandardCharsets.UTF_8))) {
            for (String line; (line = r.readLine()) != null;) {
                log.append(line).append('\n');
                if (line.startsWith("TCD-JSON:")) {
                    report = line.substring("TCD-JSON:".length());
                }
            }
        }
        int rc = p.waitFor();
        watchdog.interrupt();
        Path gdbLog = sessionDir.resolve(s == null ? "gdb-attach-" + System.currentTimeMillis() + ".log" : "gdb.log");
        Files.writeString(gdbLog, log);
        if (s != null) {
            s.close();
        }
        if (report == null) {
            err.println("tcd: no report produced (cuda-gdb exit " + rc + "). Log: " + gdbLog);
            log.toString().lines().filter(l -> !l.startsWith("[New Thread") && !l.contains("exited]")).skip(Math.max(0, log.toString().lines().count() - 25)).forEach(err::println);
            return 1;
        }
        Object doc = Json.parse(report);
        if (o.json) {
            out.println(report);
        } else {
            printBatch(Json.path(doc, "batch"), sessionDir);
        }
        return 0;
    }

    @SuppressWarnings("unchecked")
    static void printBatch(Object b, Path sessionDir) {
        List<Object> hits = (List<Object>) Json.path(b, "hits");
        if (hits == null || hits.isEmpty()) {
            out.println("No breakpoint was hit (" + Json.path(b, "exit") + "). Check the kernel name: it is the Java method name of the task.");
            Object host = Json.path(b, "hostStop");
            if (host != null) {
                out.println("Stopped in host code instead:\n" + host);
            }
            return;
        }
        Object kernels = Json.path(b, "kernels");
        if (kernels != null) {
            out.println(kernels.toString().stripTrailing());
        }
        int n = 0;
        for (Object h : hits) {
            Object w = Json.path(h, "where");
            out.printf("%n== hit %d: %s at line %s   %s%n", ++n, Json.path(w, "function"), Json.path(w, "line"), Optional.ofNullable(Json.path(w, "code")).orElse(""));
            printContext((String) Json.path(w, "source"), toInt(Json.path(w, "line")));
            Map<String, Object> arrays = (Map<String, Object>) Json.path(h, "arrays");
            if (arrays != null && !arrays.isEmpty()) {
                out.println("\n   array arguments (global memory, same for every thread):");
                arrays.forEach((k, v) -> out.printf("   %-20s = %s%n", k, v));
            }
            printThreadDifferences((List<Object>) Json.path(h, "threads"));
            for (Object t : (List<Object>) Json.path(h, "threads")) {
                Object meaning = Json.path(t, "meaning");
                out.printf("%n-- block %s thread %s  (line %s%s)%n", coord(Json.path(t, "block")), coord(Json.path(t, "thread")), Json.path(t, "line"),
                        meaning == null ? "" : ": " + meaning);
                if (Json.path(t, "error") != null) {
                    out.println("   " + Json.path(t, "error"));
                    continue;
                }
                Map<String, Object> locals = (Map<String, Object>) Json.path(t, "locals");
                Map<String, Object> printed = (Map<String, Object>) Json.path(t, "print");
                if (printed != null && !printed.isEmpty()) {
                    printed.forEach((k, v) -> out.printf("   %-20s = %s%n", k, v));
                    out.println("   ...");
                }
                Map<String, Object> shared = (Map<String, Object>) Json.path(t, "shared");
                if (shared != null) {
                    shared.forEach((k, v) -> out.printf("   __shared__ %-9s = %s%n", k, v));
                }
                Map<String, Object> hints = (Map<String, Object>) Json.path(t, "hints");
                if (locals != null) {
                    locals.entrySet().stream().filter(e -> !e.getKey().startsWith("_")).sorted(Map.Entry.comparingByKey(tcd::naturalOrder))
                            .forEach(e -> {
                                Object hint = hints == null ? null : hints.get(e.getKey());
                                String v = String.valueOf(e.getValue());
                                out.printf("   %-20s = %s%n", e.getKey(), hint == null ? v : String.format("%-24s # %s", v, hint));
                            });
                }
            }
        }
        out.println("\n(" + Json.path(b, "exit") + "; generated sources in " + sessionDir.resolve("src") + ")");
    }

    /**
     * With several threads, the locals whose values differ are the interesting ones. A loop
     * variable that differs between warps of one block at the same stop, for example, shows a missing barrier.
     */
    @SuppressWarnings("unchecked")
    static void printThreadDifferences(List<Object> threads) {
        List<Map<String, Object>> ok = new ArrayList<>();
        for (Object t : threads) {
            if (Json.path(t, "locals") instanceof Map<?, ?> m && !m.isEmpty()) {
                ok.add((Map<String, Object>) t);
            }
        }
        if (ok.size() < 2) {
            return;
        }
        List<String> names = new ArrayList<>(((Map<String, Object>) ok.get(0).get("locals")).keySet());
        names.removeIf(n -> n.startsWith("_") || n.startsWith("ul_") || n.startsWith("arg"));
        names.sort(tcd::naturalOrder);
        List<String> differing = new ArrayList<>();
        for (String n : names) {
            Object first = ((Map<String, Object>) ok.get(0).get("locals")).get(n);
            for (Map<String, Object> t : ok) {
                Object v = ((Map<String, Object>) t.get("locals")).get(n);
                if (!String.valueOf(first).equals(String.valueOf(v)) && !String.valueOf(v).startsWith("<")) {
                    differing.add(n);
                    break;
                }
            }
        }
        if (differing.isEmpty()) {
            return;
        }
        out.println("\n   locals that differ between the inspected threads:");
        StringBuilder head = new StringBuilder(String.format("   %-12s", ""));
        for (Map<String, Object> t : ok) {
            head.append(String.format(" %-16s", "b" + coord(t.get("block")).replace(",0,0)", ")").replace("(", "") + " t" + coord(t.get("thread")).replace(",0,0)", ")").replace("(", "").replace(")", "")));
        }
        out.println(head.toString().replace(")", ""));
        Map<String, Object> hints0 = (Map<String, Object>) ok.get(0).get("hints");
        for (String n : differing) {
            StringBuilder row = new StringBuilder(String.format("   %-12s", n));
            for (Map<String, Object> t : ok) {
                row.append(String.format(" %-16s", String.valueOf(((Map<String, Object>) t.get("locals")).get(n))));
            }
            Object hint = hints0 == null ? null : hints0.get(n);
            out.println(row + (hint == null ? "" : "  # " + hint));
        }
        out.println("   (line: " + String.join(", ", ok.stream().map(t -> String.valueOf(t.get("line"))).toList()) + ")");
    }

    /** Orders generated names by their numeric suffix: i_3 before l_4 before f_12. */
    static int naturalOrder(String a, String b) {
        return Integer.compare(suffix(a), suffix(b)) != 0 ? Integer.compare(suffix(a), suffix(b)) : a.compareTo(b);
    }

    static int suffix(String n) {
        Matcher m = Pattern.compile("_(\\d+)$").matcher(n);
        return m.find() ? Integer.parseInt(m.group(1)) : -1;
    }

    static void printContext(String src, int line) {
        if (src == null || line <= 0) {
            return;
        }
        try {
            List<String> lines = Files.readAllLines(Path.of(src));
            for (int i = Math.max(1, line - 3); i <= Math.min(lines.size(), line + 2); i++) {
                out.printf("%s%4d  %s%n", i == line ? " => " : "    ", i, lines.get(i - 1));
            }
        } catch (IOException ignored) {
        }
    }

    static String coord(Object o) {
        return o instanceof List<?> l ? "(" + String.join(",", l.stream().map(String::valueOf).toList()) + ")" : String.valueOf(o);
    }

    static int toInt(Object o) {
        return o instanceof Number n ? n.intValue() : -1;
    }

    // ------------------------------------------------------------------ diff

    /** tcd diff a.json b.json: which locals differ for the same thread at the same hit. */
    @SuppressWarnings("unchecked")
    static int diff(Options o) throws IOException {
        if (o.appArgs.size() != 2) {
            err.println("usage: tcd diff A.json B.json   (reports from `tcd batch --json`)");
            return 2;
        }
        Object a = Json.path(Json.parse(lastJsonLine(Path.of(o.appArgs.get(0)))), "batch");
        Object b = Json.path(Json.parse(lastJsonLine(Path.of(o.appArgs.get(1)))), "batch");
        List<Object> ha = (List<Object>) Json.path(a, "hits"), hb = (List<Object>) Json.path(b, "hits");
        int differences = 0;
        for (int i = 0; i < Math.min(ha.size(), hb.size()); i++) {
            Object wa = Json.path(ha.get(i), "where"), wb = Json.path(hb.get(i), "where");
            out.printf("== hit %d: A %s:%s   B %s:%s%n", i + 1, Json.path(wa, "function"), Json.path(wa, "line"), Json.path(wb, "function"), Json.path(wb, "line"));
            Map<String, Object> byThreadB = new LinkedHashMap<>();
            for (Object t : (List<Object>) Json.path(hb.get(i), "threads")) {
                byThreadB.put(coord(Json.path(t, "block")) + coord(Json.path(t, "thread")), t);
            }
            for (Object ta : (List<Object>) Json.path(ha.get(i), "threads")) {
                String id = coord(Json.path(ta, "block")) + coord(Json.path(ta, "thread"));
                Object tb = byThreadB.get(id);
                if (tb == null) {
                    continue;
                }
                // Same kernel: compare by generated name. Different kernels (such as buggy vs fixed) number
                // their SSA values differently, so compare by meaning instead (ctx.globalIdx, arg1[...]).
                boolean sameKernel = Objects.equals(Json.path(wa, "function"), Json.path(wb, "function"));
                Map<String, Object> la = sameKernel ? merged(ta) : byMeaning(ta), lb = sameKernel ? merged(tb) : byMeaning(tb);
                Map<String, Object> hints = sameKernel ? (Map<String, Object>) Json.path(ta, "hints") : null;
                List<String> names = new ArrayList<>(la.keySet());
                names.retainAll(lb.keySet());
                names.sort(tcd::naturalOrder);
                List<String> rows = new ArrayList<>();
                for (String n : names) {
                    String va = String.valueOf(la.get(n)), vb = String.valueOf(lb.get(n));
                    if (!va.equals(vb) && (!sameKernel || !n.startsWith("_") && !n.startsWith("ul_") && !n.startsWith("arg"))) {
                        Object hint = hints == null ? null : hints.get(n);
                        rows.add(String.format("   %-14s A=%-18s B=%-18s%s", n, va, vb, hint == null ? "" : " # " + hint));
                    }
                }
                if (sameKernel && !Objects.equals(Json.path(ta, "line"), Json.path(tb, "line"))) {
                    rows.add(0, String.format("   %-14s A=%-18s B=%s", "(line)", Json.path(ta, "line"), Json.path(tb, "line")));
                }
                out.printf("-- block %s thread %s: %s%s%n", coord(Json.path(ta, "block")), coord(Json.path(ta, "thread")), rows.isEmpty() ? "same" : rows.size() + " difference(s)",
                        sameKernel ? "" : "  (different kernels: compared by meaning)");
                rows.forEach(out::println);
                differences += rows.size();
            }
        }
        if (ha.size() != hb.size()) {
            out.printf("(A has %d hit(s), B has %d)%n", ha.size(), hb.size());
        }
        return differences == 0 ? 0 : 1;
    }

    /** Values keyed by their recovered Java meaning; the first variable with a meaning wins. */
    @SuppressWarnings("unchecked")
    static Map<String, Object> byMeaning(Object thread) {
        Map<String, Object> m = new LinkedHashMap<>();
        Map<String, Object> locals = (Map<String, Object>) Json.path(thread, "locals");
        Map<String, Object> hints = (Map<String, Object>) Json.path(thread, "hints");
        if (locals != null && hints != null) {
            hints.forEach((var, meaning) -> {
                Object v = locals.get(var);
                if (v != null && !String.valueOf(v).startsWith("<") && !String.valueOf(meaning).startsWith("base of") && !String.valueOf(meaning).startsWith("&")) {
                    m.putIfAbsent(String.valueOf(meaning), v);
                }
            });
        }
        if (Json.path(thread, "shared") instanceof Map<?, ?> sh) {
            ((Map<String, Object>) sh).forEach((k, v) -> m.put("__shared__ " + k, v));
        }
        return m;
    }

    @SuppressWarnings("unchecked")
    static Map<String, Object> merged(Object thread) {
        Map<String, Object> m = new LinkedHashMap<>();
        if (Json.path(thread, "locals") instanceof Map<?, ?> l) {
            m.putAll((Map<String, Object>) l);
        }
        if (Json.path(thread, "shared") instanceof Map<?, ?> sh) {
            ((Map<String, Object>) sh).forEach((k, v) -> m.put("__shared__ " + k, v));
        }
        if (Json.path(thread, "print") instanceof Map<?, ?> p) {
            m.putAll((Map<String, Object>) p);
        }
        return m;
    }

    static String lastJsonLine(Path p) throws IOException {
        List<String> lines = Files.readAllLines(p);
        for (int i = lines.size() - 1; i >= 0; i--) {
            String l = lines.get(i).trim();
            if (l.startsWith("{")) {
                return l.startsWith("{\"batch\"") ? l : l;
            }
            if (l.startsWith("TCD-JSON:")) {
                return l.substring(9);
            }
        }
        throw new IOException("no JSON report in " + p);
    }

    // ------------------------------------------------------------------ memcheck

    static final Pattern SANITIZER_FRAME = Pattern.compile("(\\w+)\\+0x[0-9a-fA-F]+ in \\S*tornado_kernel\\.cu:(\\d+)");

    static int memcheck(Options o) throws Exception {
        requireApp(o);
        Session s = new Session(tornadoHome(o), o.keep);
        s.elements = o.elements;
        Path san = cudaTool("compute-sanitizer", null).orElseThrow(() -> new IllegalStateException("compute-sanitizer not found"));
        List<String> tools = o.tools.isEmpty() ? List.of("memcheck") : o.tools;
        List<SanitizerSummary> summaries = new ArrayList<>();
        int appRc = 0;
        for (String tool : tools) {
            List<String> cmd = new ArrayList<>(List.of(san.toString(), "--tool", tool, "--show-backtrace", "device"));
            // -lineinfo only: -G would serialize the kernel and can hide races.
            cmd.addAll(s.javaCommand(o, "-lineinfo"));
            err.println("tcd: compute-sanitizer --tool " + tool + " (session " + s.dir + ")");
            ProcessBuilder pb = new ProcessBuilder(cmd).redirectErrorStream(true);
            pb.environment().putAll(s.env());
            Process p = pb.start();
            SanitizerSummary summary = new SanitizerSummary(tool);
            summaries.add(summary);
            Path logFile = s.dir.resolve("sanitizer-" + tool + ".log");
            summary.log = logFile;
            try (BufferedReader r = new BufferedReader(new InputStreamReader(p.getInputStream(), StandardCharsets.UTF_8));
                    Writer log = Files.newBufferedWriter(logFile)) {
                for (String line; (line = r.readLine()) != null;) {
                    Matcher m = SANITIZER_FRAME.matcher(line);
                    String location = m.find() ? m.group(1) + ":" + m.group(2) : null;
                    String code = location == null ? null : generatedLine(s, m.group(1), Integer.parseInt(m.group(2)));
                    if (location != null) {
                        // The sanitizer names NVRTC's virtual file relative to the cwd; point at the real copy.
                        line = line.replaceFirst("in \\S*tornado_kernel\\.cu:", "in " + Matcher.quoteReplacement(s.dir.resolve("src").resolve(m.group(1) + ".cu").toString()) + ":");
                    }
                    summary.accept(line, location, code);
                    log.write(line + "\n");
                    if (code != null) {
                        log.write("=========         >> " + location + "  " + code + "\n");
                    }
                    PrintStream sink = o.json ? err : out;
                    if (o.json && line.startsWith("=========")) {
                        continue; // --json: reports go to the log and the JSON summary only
                    }
                    if (o.verbose || !line.startsWith("=========") || summary.reported <= 3 || line.contains("SUMMARY")) {
                        sink.println(line);
                        if (code != null) {
                            sink.println("=========         >> " + location + "  " + code);
                        }
                    }
                }
            }
            appRc = Math.max(appRc, p.waitFor());
            if (!o.json) {
                summary.print(o.verbose);
            }
        }
        s.close();
        int errors = summaries.stream().mapToInt(x -> x.errors).sum();
        if (o.json) {
            List<Object> tj = new ArrayList<>();
            for (SanitizerSummary x : summaries) {
                tj.add(x.toJson());
            }
            Map<String, Object> doc = new LinkedHashMap<>();
            doc.put("errors", errors);
            doc.put("tools", tj);
            doc.put("applicationExitCode", appRc);
            doc.put("sources", s.dir.resolve("src").toString());
            out.println(Json.write(doc));
        } else {
            err.println("tcd: generated kernel sources: " + s.dir.resolve("src") + "/<kernel>.cu");
            err.println("tcd: " + (errors == 0 ? "no sanitizer errors" : errors + " sanitizer error(s)") + (appRc != 0 ? ", application exit code " + appRc : ""));
        }
        // CI contract: 1 = sanitizer errors, otherwise the application's own exit code.
        return errors > 0 ? 1 : appRc;
    }

    /**
     * Groups sanitizer reports by (error kind, kernel line). A missing bounds check on
     * a grid tail produces one report per thread; developers want one line per bug.
     */
    static final class SanitizerSummary {
        static final Pattern HEADER = Pattern.compile("^========= (Invalid .*|Error: .*|Warning: .*|Race reported.*|Uninitialized .*|Barrier error.*|Program hit .*)");
        static final Pattern THREAD = Pattern.compile("by thread \\((\\d+),(\\d+),(\\d+)\\) in block \\((\\d+),(\\d+),(\\d+)\\)");
        final String tool;
        final Map<String, int[]> counts = new LinkedHashMap<>();
        final Map<String, String> firstThread = new LinkedHashMap<>();
        final Map<String, String> codeOf = new LinkedHashMap<>();
        String kind;
        String key;
        int reported;
        int errors;
        int warnings;
        Path log;
        static final Pattern ERROR_SUMMARY = Pattern.compile("ERROR SUMMARY: (\\d+) error");
        static final Pattern RACE_SUMMARY = Pattern.compile("RACECHECK SUMMARY: \\d+ hazards? displayed \\((\\d+) errors?, (\\d+) warnings?\\)");

        SanitizerSummary(String tool) {
            this.tool = tool;
        }

        void accept(String line, String location, String code) {
            Matcher es = ERROR_SUMMARY.matcher(line);
            if (es.find()) {
                errors = Integer.parseInt(es.group(1));
            }
            Matcher rs = RACE_SUMMARY.matcher(line);
            if (rs.find()) {
                errors = Integer.parseInt(rs.group(1));
                warnings = Integer.parseInt(rs.group(2));
            }
            Matcher h = HEADER.matcher(line);
            if (h.find()) {
                kind = h.group(1).replaceAll(" at \\S+\\+0x[0-9a-f]+ in \\S+", "").replaceAll("0x[0-9a-f]+", "").trim();
                if (kind.length() > 70) {
                    kind = kind.substring(0, 70) + "...";
                }
                key = null;
                reported++;
            }
            if (location != null && key == null && kind != null) {
                key = kind + " @ " + location;
                counts.computeIfAbsent(key, k -> new int[1])[0]++;
                codeOf.putIfAbsent(key, code);
            }
            Matcher t = THREAD.matcher(line);
            if (t.find() && key != null) {
                firstThread.putIfAbsent(key, "block (" + t.group(4) + "," + t.group(5) + "," + t.group(6) + ") thread (" + t.group(1) + "," + t.group(2) + "," + t.group(3) + ")");
            }
        }

        Map<String, Object> toJson() {
            List<Object> groups = new ArrayList<>();
            counts.forEach((k, c) -> {
                Map<String, Object> g = new LinkedHashMap<>();
                int at = k.lastIndexOf(" @ ");
                g.put("kind", k.substring(0, at));
                g.put("location", k.substring(at + 3));
                g.put("code", codeOf.get(k));
                g.put("count", c[0]);
                g.put("first", firstThread.get(k));
                groups.add(g);
            });
            Map<String, Object> m = new LinkedHashMap<>();
            m.put("tool", tool);
            m.put("errors", errors);
            m.put("warnings", warnings);
            m.put("groups", groups);
            m.put("log", String.valueOf(log));
            return m;
        }

        void print(boolean verbose) {
            if (!verbose && reported > 3) {
                out.println("========= (" + (reported - 3) + " more reports hidden; --verbose shows all)");
            }
            if (counts.isEmpty()) {
                return;
            }
            out.println("\ntcd " + tool + " summary (grouped by generated source line):");
            counts.forEach((k, c) -> {
                out.printf("  %6d x  %s%n", c[0], k);
                if (codeOf.get(k) != null) {
                    out.println("             code:  " + codeOf.get(k));
                }
                if (firstThread.get(k) != null) {
                    out.println("             first: " + firstThread.get(k) + "   -> tcd batch -b " + k.substring(k.lastIndexOf("@ ") + 2) + "@"
                            + firstThread.get(k).replaceAll("block \\(([\\d,]+)\\) thread \\(([\\d,]+)\\)", "$1:$2").replace(",0,0", "") + " -- ...");
                }
            });
        }
    }

    static final Map<String, List<String>> SOURCE_CACHE = new ConcurrentHashMap<>();

    static String generatedLine(Session s, String kernel, int line) {
        List<String> lines = SOURCE_CACHE.computeIfAbsent(kernel, k -> {
            Path f = findDump(s.dumpAbs, k);
            try {
                return f == null ? List.of() : Files.readAllLines(f);
            } catch (IOException e) {
                return List.of();
            }
        });
        return line > 0 && line <= lines.size() ? lines.get(line - 1).trim() : null;
    }

    static Path findDump(Path root, String kernel) {
        if (!Files.isDirectory(root)) {
            return null;
        }
        try (Stream<Path> st = Files.walk(root)) {
            return st.filter(p -> {
                String n = p.getFileName().toString();
                return n.endsWith("-" + kernel + ".cl") || n.equals(kernel + ".cl");
            }).max(Comparator.comparingLong(p -> p.toFile().lastModified())).orElse(null);
        } catch (IOException e) {
            return null;
        }
    }

    static List<String> dumpedKernels(Path root) {
        if (!Files.isDirectory(root)) {
            return List.of();
        }
        try (Stream<Path> st = Files.walk(root)) {
            return st.map(p -> p.getFileName().toString()).filter(n -> n.endsWith(".cl")).map(n -> n.substring(n.lastIndexOf('-') + 1, n.length() - 3)).distinct().sorted()
                    .toList();
        } catch (IOException e) {
            return List.of();
        }
    }

    // ------------------------------------------------------------------ ui (GDB/MI + HTTP)

    static int ui(Options o) throws Exception {
        requireApp(o);
        Session s = new Session(tornadoHome(o), o.keep);
        s.elements = o.elements;
        List<String> cmd = gdbBase(o, true);
        cmd.add("--args");
        cmd.addAll(s.javaCommand(o, "-G -lineinfo"));
        Mi mi = new Mi(cmd, s.env(), s.dir.resolve("mi.log"));
        mi.command("-gdb-set mi-async on");
        for (String b : o.breaks) {
            mi.console("tcd-break " + b);
        }
        Web web = new Web(mi, s, o.port);
        String url = "http://127.0.0.1:" + o.port + "/";
        err.println("tcd: web debugger at " + url + "   (session " + s.dir + ")");
        err.println("tcd: remote GPU box? ssh -L " + o.port + ":127.0.0.1:" + o.port + " <host>, then open the URL locally");
        if (o.open) {
            which("xdg-open").or(() -> which("open")).ifPresent(b -> {
                if (System.getenv("DISPLAY") != null || System.getenv("WAYLAND_DISPLAY") != null || System.getProperty("os.name").startsWith("Mac")) {
                    exec(List.of(b.toString(), url), Map.of());
                }
            });
        }
        if (!o.noRun) {
            mi.command("-exec-run");
        }
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            mi.destroy();
            s.close();
        }));
        int rc = mi.waitFor();
        web.stop();
        return rc;
    }

    /** Minimal GDB/MI driver: token-matched results, async records fanned out to listeners. */
    static final class Mi {
        record Result(String klass, Object value, String console) {
            boolean ok() {
                return !"error".equals(klass);
            }

            String message() {
                return String.valueOf(Json.path(value, "msg"));
            }
        }

        interface Listener {
            void event(String type, Object payload);
        }

        final Process proc;
        final Writer in;
        final AtomicInteger token = new AtomicInteger(1);
        final Map<Integer, CompletableFuture<Result>> pending = new ConcurrentHashMap<>();
        final Map<Integer, StringBuilder> consoleByToken = new ConcurrentHashMap<>();
        final List<Listener> listeners = new CopyOnWriteArrayList<>();
        final Object sendLock = new Object();
        volatile Integer activeToken;
        volatile String state = "idle";
        final Writer log;

        Mi(List<String> cmd, Map<String, String> env, Path logFile) throws IOException {
            ProcessBuilder pb = new ProcessBuilder(cmd);
            pb.environment().putAll(env);
            pb.redirectErrorStream(true);
            proc = pb.start();
            in = new java.io.OutputStreamWriter(proc.getOutputStream(), StandardCharsets.UTF_8);
            log = Files.newBufferedWriter(logFile);
            Thread t = new Thread(this::readLoop, "mi-reader");
            t.setDaemon(true);
            t.start();
        }

        void on(Listener l) {
            listeners.add(l);
        }

        void fire(String type, Object payload) {
            for (Listener l : listeners) {
                l.event(type, payload);
            }
        }

        /** Sends an MI command and waits for its result record. */
        Result command(String mi) {
            int t = token.getAndIncrement();
            CompletableFuture<Result> f = new CompletableFuture<>();
            pending.put(t, f);
            consoleByToken.put(t, new StringBuilder());
            synchronized (sendLock) {
                try {
                    activeToken = t;
                    in.write(t + mi + "\n");
                    in.flush();
                    logLine(">> " + t + mi);
                } catch (IOException e) {
                    return new Result("error", Map.of("msg", "cuda-gdb is not running"), "");
                }
            }
            try {
                return f.get(120, TimeUnit.SECONDS);
            } catch (Exception e) {
                pending.remove(t);
                return new Result("error", Map.of("msg", "timeout waiting for cuda-gdb"), "");
            }
        }

        /** Runs a CLI command; its console output is returned in Result.console. */
        Result console(String cli) {
            return command("-interpreter-exec console " + Json.quote(cli));
        }

        void readLoop() {
            try (BufferedReader r = new BufferedReader(new InputStreamReader(proc.getInputStream(), StandardCharsets.UTF_8))) {
                for (String line; (line = r.readLine()) != null;) {
                    logLine(line);
                    handle(line);
                }
            } catch (IOException ignored) {
            }
            state = "gone";
            fire("exited", Map.of("reason", "cuda-gdb terminated"));
            pending.values().forEach(f -> f.complete(new Result("error", Map.of("msg", "cuda-gdb terminated"), "")));
        }

        void handle(String line) {
            if (line.equals("(gdb) ") || line.equals("(gdb)")) {
                return;
            }
            Matcher m = Pattern.compile("^(\\d*)([\\^*=~@&])(.*)$").matcher(line);
            if (!m.matches()) {
                fire("program", line); // inferior output (the Java program's stdout)
                return;
            }
            String tok = m.group(1);
            char kind = m.group(2).charAt(0);
            String rest = m.group(3);
            switch (kind) {
                case '~', '@', '&' -> {
                    String text = MiParser.cstring(rest);
                    Integer at = activeToken;
                    if (kind == '~' && at != null && pending.containsKey(at)) {
                        consoleByToken.get(at).append(text);
                    }
                    if (kind != '&' || !text.startsWith("-")) {
                        fire(kind == '@' ? "program" : "console", text);
                    }
                }
                case '^' -> {
                    int comma = rest.indexOf(',');
                    String klass = comma < 0 ? rest : rest.substring(0, comma);
                    Object value = comma < 0 ? Map.of() : MiParser.results(rest.substring(comma + 1));
                    if (klass.equals("running")) {
                        state = "running";
                    }
                    if (!tok.isEmpty()) {
                        int t = Integer.parseInt(tok);
                        CompletableFuture<Result> f = pending.remove(t);
                        StringBuilder c = consoleByToken.remove(t);
                        if (f != null) {
                            f.complete(new Result(klass, value, c == null ? "" : c.toString()));
                        }
                    }
                }
                case '*' -> {
                    int comma = rest.indexOf(',');
                    String klass = comma < 0 ? rest : rest.substring(0, comma);
                    Object value = comma < 0 ? Map.of() : MiParser.results(rest.substring(comma + 1));
                    if (klass.equals("stopped")) {
                        String reason = String.valueOf(Json.path(value, "reason"));
                        state = reason.startsWith("exited") ? "exited" : "stopped";
                        fire(state.equals("exited") ? "exited" : "stopped", value);
                    } else if (klass.equals("running")) {
                        state = "running";
                        fire("running", value);
                    }
                }
                case '=' -> fire("notify", rest);
                default -> {
                }
            }
        }

        synchronized void logLine(String s) {
            try {
                log.write(s + "\n");
                log.flush();
            } catch (IOException ignored) {
            }
        }

        void interrupt() {
            // -exec-interrupt needs mi-async; SIGINT to cuda-gdb is the robust fallback.
            Result r = command("-exec-interrupt --all");
            if (!r.ok()) {
                exec(List.of("kill", "-INT", String.valueOf(proc.pid())), Map.of());
            }
        }

        int waitFor() throws InterruptedException {
            return proc.waitFor();
        }

        void destroy() {
            if (proc.isAlive()) {
                try {
                    in.write("-gdb-exit\n");
                    in.flush();
                    proc.waitFor(3, TimeUnit.SECONDS);
                } catch (Exception ignored) {
                }
                proc.destroyForcibly();
            }
        }
    }

    /** GDB/MI output syntax: c-strings, tuples {..}, lists [..] of values or name=value results. */
    static final class MiParser {
        final String s;
        int i;

        MiParser(String s) {
            this.s = s;
        }

        static Map<String, Object> results(String s) {
            MiParser p = new MiParser(s);
            Map<String, Object> m = new LinkedHashMap<>();
            while (p.i < p.s.length()) {
                String k = p.name();
                p.expect('=');
                m.put(k, p.value());
                if (p.i < p.s.length() && p.s.charAt(p.i) == ',') {
                    p.i++;
                }
            }
            return m;
        }

        static String cstring(String s) {
            return s.startsWith("\"") ? (String) new MiParser(s).value() : s;
        }

        String name() {
            int start = i;
            while (i < s.length() && s.charAt(i) != '=') {
                i++;
            }
            return s.substring(start, i);
        }

        void expect(char c) {
            if (i >= s.length() || s.charAt(i) != c) {
                throw new IllegalStateException("MI parse: expected '" + c + "' at " + i + " in " + s);
            }
            i++;
        }

        Object value() {
            char c = s.charAt(i);
            if (c == '"') {
                i++;
                StringBuilder b = new StringBuilder();
                while (i < s.length()) {
                    char ch = s.charAt(i++);
                    if (ch == '"') {
                        break;
                    }
                    if (ch == '\\' && i < s.length()) {
                        char e = s.charAt(i++);
                        switch (e) {
                            case 'n' -> b.append('\n');
                            case 't' -> b.append('\t');
                            case 'r' -> b.append('\r');
                            case '"' -> b.append('"');
                            case '\\' -> b.append('\\');
                            default -> {
                                if (e >= '0' && e <= '7') {
                                    int v = e - '0', n = 1;
                                    while (n < 3 && i < s.length() && s.charAt(i) >= '0' && s.charAt(i) <= '7') {
                                        v = v * 8 + (s.charAt(i++) - '0');
                                        n++;
                                    }
                                    b.append((char) v);
                                } else {
                                    b.append(e);
                                }
                            }
                        }
                    } else {
                        b.append(ch);
                    }
                }
                return b.toString();
            }
            if (c == '{') {
                i++;
                Map<String, Object> m = new LinkedHashMap<>();
                while (s.charAt(i) != '}') {
                    String k = name();
                    expect('=');
                    m.put(k, value());
                    if (s.charAt(i) == ',') {
                        i++;
                    }
                }
                i++;
                return m;
            }
            if (c == '[') {
                i++;
                List<Object> l = new ArrayList<>();
                while (s.charAt(i) != ']') {
                    int eq = s.indexOf('=', i);
                    int q = s.indexOf('"', i), br = s.indexOf('{', i), sq = s.indexOf('[', i);
                    boolean isResult = eq > 0 && s.charAt(i) != '"' && s.charAt(i) != '{' && s.charAt(i) != '[' && (q < 0 || eq < q) && (br < 0 || eq < br) && (sq < 0 || eq < sq);
                    if (isResult) {
                        String k = name();
                        expect('=');
                        l.add(Map.of(k, value()));
                    } else {
                        l.add(value());
                    }
                    if (s.charAt(i) == ',') {
                        i++;
                    }
                }
                i++;
                return l;
            }
            throw new IllegalStateException("MI parse: unexpected '" + c + "' at " + i);
        }
    }

    /** Local HTTP server for the web UI. Bound to 127.0.0.1 only. */
    static final class Web {
        final HttpServer server;
        final Mi mi;
        final Session session;
        final List<OutputStream> sse = new CopyOnWriteArrayList<>();
        final List<String> history = new CopyOnWriteArrayList<>(); // replayed to late SSE clients

        Web(Mi mi, Session session, int port) throws IOException {
            this.mi = mi;
            this.session = session;
            server = HttpServer.create(new InetSocketAddress(InetAddress.getLoopbackAddress(), port), 0);
            server.setExecutor(java.util.concurrent.Executors.newCachedThreadPool());
            server.createContext("/", this::statik);
            server.createContext("/api/events", this::events);
            server.createContext("/api/state", ex -> json(ex, state()));
            server.createContext("/api/exec", ex -> {
                Map<String, Object> body = body(ex);
                Mi.Result r = mi.console(String.valueOf(body.get("cmd")));
                json(ex, Map.of("ok", r.ok(), "output", r.ok() ? r.console() : r.message()));
            });
            server.createContext("/api/control", ex -> {
                String action = String.valueOf(body(ex).get("action"));
                Mi.Result r = switch (action) {
                    case "run" -> mi.command("-exec-run");
                    case "continue" -> mi.command("-exec-continue");
                    case "next" -> mi.command("-exec-next");
                    case "step" -> mi.command("-exec-step");
                    case "finish" -> mi.command("-exec-finish");
                    case "interrupt" -> {
                        mi.interrupt();
                        yield new Mi.Result("done", Map.of(), "");
                    }
                    case "kill" -> mi.command("-exec-abort");
                    default -> new Mi.Result("error", Map.of("msg", "unknown action " + action), "");
                };
                json(ex, Map.of("ok", r.ok(), "message", r.ok() ? "" : r.message()));
            });
            server.createContext("/api/break", ex -> {
                Map<String, Object> b = body(ex);
                Mi.Result r;
                if (b.containsKey("delete")) {
                    r = mi.command("-break-delete " + ((Number) b.get("delete")).intValue());
                } else {
                    String spec = b.get("spec") != null ? String.valueOf(b.get("spec"))
                            : b.get("kernel") + (b.get("line") != null ? ":" + ((Number) b.get("line")).intValue() : "");
                    r = mi.console("tcd-break " + spec);
                }
                json(ex, Map.of("ok", r.ok(), "message", r.ok() ? r.console() : r.message(), "breakpoints", breakpoints()));
            });
            server.createContext("/api/focus", ex -> {
                Map<String, Object> b = body(ex);
                Mi.Result r = mi.console("cuda block (" + b.get("block") + ") thread (" + b.get("thread") + ")");
                json(ex, Map.of("ok", r.ok(), "message", r.ok() ? r.console() : r.message()));
            });
            server.createContext("/api/array", ex -> {
                Map<String, Object> b = body(ex);
                Mi.Result r = mi.console("tcd-array --json " + b.get("expr") + " " + Json.quote(String.valueOf(b.get("type"))) + " " + num(b.get("start"), 0) + " " + num(b.get("count"), 16));
                json(ex, r.ok() ? jsonFrom(r.console(), "array") : Map.of("error", r.message()));
            });
            server.createContext("/api/source", ex -> {
                String q = Optional.ofNullable(ex.getRequestURI().getQuery()).orElse("");
                String kernel = q.startsWith("kernel=") ? java.net.URLDecoder.decode(q.substring(7), StandardCharsets.UTF_8) : "";
                Path f = findDump(session.dumpAbs, kernel);
                json(ex, Map.of("kernel", kernel, "text", f == null ? "" : Files.readString(f)));
            });
            mi.on((type, payload) -> broadcast(type, payload));
            server.start();
        }

        static int num(Object o, int def) {
            return o instanceof Number n ? n.intValue() : o == null ? def : Integer.parseInt(o.toString());
        }

        Map<String, Object> state() {
            Map<String, Object> st = new LinkedHashMap<>();
            st.put("state", mi.state);
            st.put("session", session.dir.toString());
            st.put("kernels", dumpedKernels(session.dumpAbs));
            st.put("breakpoints", breakpoints());
            if ("stopped".equals(mi.state)) {
                Mi.Result w = mi.console("tcd-where");
                Object where = jsonFrom(w.console(), "where");
                st.put("where", where);
                if (Boolean.TRUE.equals(Json.path(where, "device"))) {
                    st.put("threads", jsonFrom(mi.console("tcd-snapshot").console(), "snapshot"));
                    Object srcPath = Json.path(where, "source");
                    if (srcPath != null) {
                        st.put("arrays", jsonFrom(mi.console("python print('TCD-JSON:' + json.dumps({'arrays': array_args(analyze_source(" + Json.quote(srcPath.toString())
                                + ")[1])}))").console(), "arrays"));
                    }
                    st.put("cudaKernels", mi.console("info cuda kernels").console());
                    st.put("warps", mi.console("info cuda warps").console());
                    Object src = Json.path(where, "source");
                    if (src != null) {
                        try {
                            st.put("source", Files.readString(Path.of(src.toString())));
                        } catch (IOException ignored) {
                        }
                    }
                } else {
                    st.put("backtrace", mi.console("bt 10").console());
                }
            }
            return st;
        }

        Object breakpoints() {
            return jsonFrom(mi.console("tcd-breakpoints").console(), "breakpoints");
        }

        static Object jsonFrom(String console, String key) {
            for (String l : console.split("\n")) {
                int i = l.indexOf("TCD-JSON:");
                if (i >= 0) {
                    return Json.path(Json.parse(l.substring(i + 9)), key);
                }
            }
            return Map.of("error", console.trim());
        }

        void broadcast(String type, Object payload) {
            String msg = "data: " + Json.write(Map.of("type", type, "payload", payload)) + "\n\n";
            if (!type.equals("notify")) {
                history.add(msg);
                if (history.size() > 500) {
                    history.remove(0);
                }
            }
            for (OutputStream o : sse) {
                try {
                    o.write(msg.getBytes(StandardCharsets.UTF_8));
                    o.flush();
                } catch (IOException e) {
                    sse.remove(o);
                }
            }
        }

        void events(HttpExchange ex) throws IOException {
            ex.getResponseHeaders().add("Content-Type", "text/event-stream");
            ex.getResponseHeaders().add("Cache-Control", "no-cache");
            ex.sendResponseHeaders(200, 0);
            OutputStream o = ex.getResponseBody();
            for (String h : history) {
                o.write(h.getBytes(StandardCharsets.UTF_8));
            }
            o.flush();
            sse.add(o);
        }

        void statik(HttpExchange ex) throws IOException {
            String p = ex.getRequestURI().getPath();
            String name = p.equals("/") ? "index.html" : p.substring(1);
            if (!name.matches("[a-z]+\\.(html|js|css)")) {
                ex.sendResponseHeaders(404, -1);
                return;
            }
            byte[] data = Files.readAllBytes(resource("web/" + name));
            String type = name.endsWith(".html") ? "text/html" : name.endsWith(".js") ? "text/javascript" : "text/css";
            ex.getResponseHeaders().add("Content-Type", type + "; charset=utf-8");
            ex.sendResponseHeaders(200, data.length);
            ex.getResponseBody().write(data);
            ex.close();
        }

        @SuppressWarnings("unchecked")
        static Map<String, Object> body(HttpExchange ex) throws IOException {
            String s = new String(ex.getRequestBody().readAllBytes(), StandardCharsets.UTF_8);
            return s.isBlank() ? Map.of() : (Map<String, Object>) Json.parse(s);
        }

        static void json(HttpExchange ex, Object o) throws IOException {
            byte[] data = Json.write(o).getBytes(StandardCharsets.UTF_8);
            ex.getResponseHeaders().add("Content-Type", "application/json");
            ex.sendResponseHeaders(200, data.length);
            ex.getResponseBody().write(data);
            ex.close();
        }

        void stop() {
            server.stop(0);
        }
    }

    // ------------------------------------------------------------------ DAP (IDE integration)

    /**
     * Debug Adapter Protocol server over stdin/stdout, backed by the same GDB/MI driver as the web UI.
     * One DAP "thread" represents the CUDA focus. Switch it from the debug console with
     * `cuda block (x,y,z) thread (x,y,z)`. Breakpoints go on the generated sources, which the adapter
     * reports as sessions/<ts>/src/<kernel>/tornado_kernel.cu, with or without a condition.
     */
    static final class Dap {
        final Options opts;
        final OutputStream wire;
        final AtomicInteger seq = new AtomicInteger(1);
        Mi mi;
        Session session;
        final Map<String, List<Integer>> bpsBySource = new ConcurrentHashMap<>();
        final Map<Integer, Object> varRefs = new ConcurrentHashMap<>();
        final AtomicInteger nextRef = new AtomicInteger(100);
        volatile Object lastWhere;
        volatile boolean terminated;

        Dap(Options o, OutputStream wire) {
            this.opts = o;
            this.wire = wire;
        }

        static int serve(Options o) throws Exception {
            OutputStream wire = new java.io.FileOutputStream(java.io.FileDescriptor.out);
            System.setOut(System.err); // stdout carries only DAP messages; anything else goes to stderr
            Dap d = new Dap(o, wire);
            d.loop(new java.io.BufferedInputStream(System.in));
            return 0;
        }

        void loop(InputStream in) throws IOException {
            while (!terminated) {
                int length = -1;
                String header;
                while ((header = readLine(in)) != null && !header.isEmpty()) {
                    if (header.toLowerCase().startsWith("content-length:")) {
                        length = Integer.parseInt(header.substring(15).trim());
                    }
                }
                if (header == null || length < 0) {
                    break;
                }
                byte[] body = in.readNBytes(length);
                Object msg = Json.parse(new String(body, StandardCharsets.UTF_8));
                try {
                    handle(msg);
                } catch (Exception e) {
                    respond(msg, false, e.toString(), null);
                }
            }
            if (mi != null) {
                mi.destroy();
            }
            if (session != null) {
                session.close();
            }
        }

        static String readLine(InputStream in) throws IOException {
            StringBuilder b = new StringBuilder();
            for (int c; (c = in.read()) != -1;) {
                if (c == '\n') {
                    return b.toString().replace("\r", "");
                }
                b.append((char) c);
            }
            return b.isEmpty() ? null : b.toString();
        }

        synchronized void send(Map<String, Object> m) {
            m.put("seq", seq.getAndIncrement());
            byte[] body = Json.write(m).getBytes(StandardCharsets.UTF_8);
            try {
                wire.write(("Content-Length: " + body.length + "\r\n\r\n").getBytes(StandardCharsets.US_ASCII));
                wire.write(body);
                wire.flush();
            } catch (IOException e) {
                terminated = true;
            }
        }

        void respond(Object req, boolean ok, String message, Object body) {
            Map<String, Object> m = new LinkedHashMap<>();
            m.put("type", "response");
            m.put("request_seq", Json.path(req, "seq"));
            m.put("success", ok);
            m.put("command", Json.path(req, "command"));
            if (message != null) {
                m.put("message", message);
            }
            if (body != null) {
                m.put("body", body);
            }
            send(m);
        }

        void event(String name, Object body) {
            Map<String, Object> m = new LinkedHashMap<>();
            m.put("type", "event");
            m.put("event", name);
            if (body != null) {
                m.put("body", body);
            }
            send(m);
        }

        void output(String category, String text) {
            event("output", Map.of("category", category, "output", text.endsWith("\n") ? text : text + "\n"));
        }

        @SuppressWarnings("unchecked")
        void handle(Object req) throws Exception {
            String cmd = String.valueOf(Json.path(req, "command"));
            Map<String, Object> args = Json.path(req, "arguments") instanceof Map<?, ?> a ? (Map<String, Object>) a : Map.of();
            switch (cmd) {
                case "initialize" -> respond(req, true, null, Map.of("supportsConfigurationDoneRequest", true, "supportsConditionalBreakpoints", true,
                        "supportsEvaluateForHovers", true, "supportsTerminateRequest", true));
                case "launch", "attach" -> {
                    launch(cmd, args);
                    respond(req, true, null, null);
                    event("initialized", null);
                }
                case "setBreakpoints" -> respond(req, true, null, Map.of("breakpoints", setBreakpoints(args)));
                case "setExceptionBreakpoints", "setFunctionBreakpoints" -> respond(req, true, null, Map.of("breakpoints", List.of()));
                case "configurationDone" -> {
                    respond(req, true, null, null);
                    if (!"attach".equals(mode)) {
                        mi.command("-exec-run");
                    } else {
                        mi.command("-exec-continue");
                    }
                }
                case "threads" -> respond(req, true, null, Map.of("threads", List.of(Map.of("id", 1, "name", threadName()))));
                case "stackTrace" -> respond(req, true, null, stackTrace());
                case "scopes" -> respond(req, true, null, Map.of("scopes", scopes()));
                case "variables" -> respond(req, true, null, Map.of("variables", variables(((Number) args.get("variablesReference")).intValue())));
                case "continue" -> {
                    mi.command("-exec-continue");
                    respond(req, true, null, Map.of("allThreadsContinued", true));
                }
                case "next" -> control(req, "-exec-next");
                case "stepIn" -> control(req, "-exec-step");
                case "stepOut" -> control(req, "-exec-finish");
                case "pause" -> {
                    mi.interrupt();
                    respond(req, true, null, null);
                }
                case "evaluate" -> respond(req, true, null, evaluate(String.valueOf(args.get("expression")), String.valueOf(args.get("context"))));
                case "disconnect", "terminate" -> {
                    if ("attach".equals(mode)) {
                        mi.console("detach");
                    }
                    respond(req, true, null, null);
                    terminated = true;
                    event("terminated", null);
                }
                default -> respond(req, false, "unsupported request " + cmd, null);
            }
        }

        String mode = "launch";

        /** launch: {args: [java args], tornadoHome?, stopOnKernel?: [..], noFastStart?}; attach: {pid}. */
        @SuppressWarnings("unchecked")
        void launch(String how, Map<String, Object> args) throws Exception {
            mode = how;
            Options o = opts;
            if (args.get("tornadoHome") != null) {
                o.tornadoHome = String.valueOf(args.get("tornadoHome"));
            }
            if (Boolean.TRUE.equals(args.get("noFastStart"))) {
                o.fastStart = false;
            }
            List<String> cmd = gdbBase(o, true);
            Map<String, String> env;
            if (how.equals("attach")) {
                long pid = ((Number) args.get("pid")).longValue();
                env = launchedEnv(pid);
                cmd.addAll(List.of("-p", String.valueOf(pid)));
            } else {
                o.appArgs = new ArrayList<>((List<String>) args.getOrDefault("args", List.of()));
                session = new Session(tornadoHome(o), o.keep);
                env = session.env();
                cmd.add("--args");
                cmd.addAll(session.javaCommand(o, "-G -lineinfo"));
            }
            Path sessionDir = Path.of(env.get("TCD_SESSION"));
            mi = new Mi(cmd, env, sessionDir.resolve("dap-mi.log"));
            mi.command("-gdb-set mi-async on");
            for (Object k : (List<Object>) args.getOrDefault("stopOnKernel", List.of())) {
                mi.console("tcd-break " + k);
            }
            mi.on((type, payload) -> {
                switch (type) {
                    case "stopped" -> {
                        varRefs.clear();
                        String reason = String.valueOf(Json.path(payload, "reason"));
                        event("stopped", Map.of("reason", reason.contains("breakpoint") ? "breakpoint" : reason.contains("step") || reason.contains("end-stepping") ? "step" : "pause",
                                "threadId", 1, "allThreadsStopped", true));
                    }
                    case "running" -> event("continued", Map.of("threadId", 1, "allThreadsContinued", true));
                    case "exited" -> {
                        event("exited", Map.of("exitCode", toInt(Json.path(payload, "exit-code")) < 0 ? 0 : toInt(Json.path(payload, "exit-code"))));
                        event("terminated", null);
                    }
                    case "program" -> output("stdout", String.valueOf(payload));
                    case "console" -> {
                        String t = String.valueOf(payload);
                        if (!t.startsWith("TCD-JSON:") && !t.startsWith("[New Thread") && !t.contains("exited]")) {
                            output("console", t);
                        }
                    }
                    default -> {
                    }
                }
            });
            output("console", "tcd: session " + sessionDir + ". Switch GPU thread in the debug console: cuda block (x,y,z) thread (x,y,z)");
        }

        void control(Object req, String mic) {
            Mi.Result r = mi.command(mic);
            respond(req, r.ok(), r.ok() ? null : r.message(), null);
        }

        /** Kernel of a source path: .../src/<kernel>/tornado_kernel.cu or .../<kernel>.cu */
        static String kernelOf(String path) {
            Path p = Path.of(path);
            String f = p.getFileName().toString();
            if (f.equals("tornado_kernel.cu") && p.getParent() != null) {
                return p.getParent().getFileName().toString();
            }
            return f.replaceFirst("\\.(cu|cl)$", "").replaceFirst("^.*-", "");
        }

        @SuppressWarnings("unchecked")
        List<Object> setBreakpoints(Map<String, Object> args) {
            String path = String.valueOf(Json.path(args.get("source"), "path"));
            String kernel = kernelOf(path);
            for (Integer n : bpsBySource.getOrDefault(path, List.of())) {
                mi.command("-break-delete " + n);
            }
            List<Integer> numbers = new ArrayList<>();
            List<Object> result = new ArrayList<>();
            for (Object b : (List<Object>) args.getOrDefault("breakpoints", List.of())) {
                int line = toInt(Json.path(b, "line"));
                Object cond = Json.path(b, "condition");
                Mi.Result r = mi.console("tcd-break " + kernel + ":" + line + (cond == null || cond.toString().isBlank() ? "" : " if " + cond));
                Matcher m = Pattern.compile("breakpoint (\\d+)").matcher(r.console());
                boolean ok = r.ok() && m.find();
                if (ok) {
                    numbers.add(Integer.parseInt(m.group(1)));
                }
                Map<String, Object> v = new LinkedHashMap<>();
                v.put("verified", ok);
                v.put("line", line);
                v.put("message", ok ? "kernel " + kernel + " (resolves when the kernel is JIT-compiled)" : r.message());
                result.add(v);
            }
            bpsBySource.put(path, numbers);
            return result;
        }

        String threadName() {
            Object f = Json.path(lastWhere, "focus");
            return f == null ? "GPU (CUDA focus)" : "GPU block " + coord(Json.path(f, "block")) + " thread " + coord(Json.path(f, "thread"));
        }

        Map<String, Object> stackTrace() {
            Object w = Web.jsonFrom(mi.console("tcd-where").console(), "where");
            lastWhere = w;
            List<Object> frames = new ArrayList<>();
            if (Boolean.TRUE.equals(Json.path(w, "device"))) {
                String kernel = String.valueOf(Json.path(w, "function"));
                Map<String, Object> f = new LinkedHashMap<>();
                f.put("id", 1);
                f.put("name", kernel + " @ block " + coord(Json.path(Json.path(w, "focus"), "block")) + " thread " + coord(Json.path(Json.path(w, "focus"), "thread")));
                f.put("line", toInt(Json.path(w, "line")));
                f.put("column", 1);
                if (Json.path(w, "source") != null) {
                    f.put("source", Map.of("name", kernel + ".cu", "path", String.valueOf(Json.path(w, "source"))));
                }
                frames.add(f);
            } else {
                frames.add(Map.of("id", 1, "name", "host (JVM) - not in a kernel", "line", 0, "column", 0));
            }
            return Map.of("stackFrames", frames, "totalFrames", frames.size());
        }

        List<Object> scopes() {
            Object snap = Web.jsonFrom(mi.console("tcd-snapshot").console(), "snapshot");
            Object thread = snap instanceof List<?> l && !l.isEmpty() ? l.get(0) : Map.of();
            int locals = ref(Map.of("kind", "locals", "thread", thread));
            int shared = ref(Map.of("kind", "shared", "thread", thread));
            int cuda = ref(Map.of("kind", "cuda"));
            return List.of(Map.of("name", "Locals", "variablesReference", locals, "expensive", false),
                    Map.of("name", "__shared__", "variablesReference", shared, "expensive", false),
                    Map.of("name", "Arrays / CUDA", "variablesReference", cuda, "expensive", true));
        }

        int ref(Object o) {
            int r = nextRef.getAndIncrement();
            varRefs.put(r, o);
            return r;
        }

        @SuppressWarnings("unchecked")
        List<Object> variables(int ref) {
            Object scope = varRefs.get(ref);
            List<Object> out = new ArrayList<>();
            String kind = String.valueOf(Json.path(scope, "kind"));
            if (kind.equals("locals") || kind.equals("shared")) {
                Object t = Json.path(scope, "thread");
                Map<String, Object> vals = (Map<String, Object>) Json.path(t, kind);
                Map<String, Object> hints = (Map<String, Object>) Json.path(t, "hints");
                if (vals != null) {
                    List<String> names = new ArrayList<>(vals.keySet());
                    names.sort(tcd::naturalOrder);
                    for (String n : names) {
                        Object hint = hints == null ? null : hints.get(n);
                        Map<String, Object> v = new LinkedHashMap<>();
                        v.put("name", n);
                        v.put("value", String.valueOf(vals.get(n)) + (hint == null ? "" : "    ← " + hint));
                        v.put("variablesReference", 0);
                        if (hint != null) {
                            v.put("type", String.valueOf(hint));
                        }
                        out.add(v);
                    }
                }
            } else if (kind.equals("cuda")) {
                String where = String.valueOf(Json.path(lastWhere, "source"));
                Mi.Result arrays = mi.console("python print('TCD-JSON:' + json.dumps({'arrays': array_args(analyze_source(" + Json.quote(where) + ")[1])}))");
                Object a = Web.jsonFrom(arrays.console(), "arrays");
                if (a instanceof Map<?, ?> m) {
                    m.forEach((k, v) -> out.add(Map.of("name", String.valueOf(k), "value", String.valueOf(v), "variablesReference", 0)));
                }
                out.add(Map.of("name", "info cuda kernels", "value", mi.console("info cuda kernels").console().trim(), "variablesReference", 0));
            }
            return out;
        }

        Map<String, Object> evaluate(String expr, String context) {
            if ("repl".equals(context)) {
                Mi.Result r = mi.console(expr);
                if (expr.startsWith("cuda ") && r.ok()) {
                    // Focus changed: refresh the IDE's views.
                    varRefs.clear();
                    event("stopped", Map.of("reason", "focus", "threadId", 1, "allThreadsStopped", true, "description", "CUDA focus changed"));
                }
                return Map.of("result", r.ok() ? r.console().stripTrailing() : r.message(), "variablesReference", 0);
            }
            Mi.Result r = mi.command("-data-evaluate-expression " + Json.quote(expr));
            return Map.of("result", r.ok() ? String.valueOf(Json.path(r.value(), "value")) : r.message(), "variablesReference", 0);
        }
    }

    // ------------------------------------------------------------------ tiny JSON

    static final class Json {
        static Object path(Object o, String key) {
            return o instanceof Map<?, ?> m ? m.get(key) : null;
        }

        static String quote(String s) {
            StringBuilder b = new StringBuilder("\"");
            for (char c : s.toCharArray()) {
                switch (c) {
                    case '"' -> b.append("\\\"");
                    case '\\' -> b.append("\\\\");
                    case '\n' -> b.append("\\n");
                    case '\r' -> b.append("\\r");
                    case '\t' -> b.append("\\t");
                    default -> {
                        if (c < 0x20) {
                            b.append(String.format("\\u%04x", (int) c));
                        } else {
                            b.append(c);
                        }
                    }
                }
            }
            return b.append('"').toString();
        }

        static String write(Object o) {
            if (o == null) {
                return "null";
            }
            if (o instanceof String s) {
                return quote(s);
            }
            if (o instanceof Number || o instanceof Boolean) {
                return o.toString();
            }
            if (o instanceof Map<?, ?> m) {
                StringBuilder b = new StringBuilder("{");
                m.forEach((k, v) -> b.append(b.length() > 1 ? "," : "").append(quote(String.valueOf(k))).append(':').append(write(v)));
                return b.append('}').toString();
            }
            if (o instanceof List<?> l) {
                StringBuilder b = new StringBuilder("[");
                l.forEach(v -> b.append(b.length() > 1 ? "," : "").append(write(v)));
                return b.append(']').toString();
            }
            return quote(o.toString());
        }

        static Object parse(String s) {
            int[] i = { 0 };
            return value(s, i);
        }

        static void ws(String s, int[] i) {
            while (i[0] < s.length() && Character.isWhitespace(s.charAt(i[0]))) {
                i[0]++;
            }
        }

        static Object value(String s, int[] i) {
            ws(s, i);
            char c = s.charAt(i[0]);
            switch (c) {
                case '{' -> {
                    i[0]++;
                    Map<String, Object> m = new LinkedHashMap<>();
                    ws(s, i);
                    if (s.charAt(i[0]) == '}') {
                        i[0]++;
                        return m;
                    }
                    while (true) {
                        String k = (String) value(s, i);
                        ws(s, i);
                        i[0]++; // ':'
                        m.put(k, value(s, i));
                        ws(s, i);
                        if (s.charAt(i[0]++) == '}') {
                            return m;
                        }
                    }
                }
                case '[' -> {
                    i[0]++;
                    List<Object> l = new ArrayList<>();
                    ws(s, i);
                    if (s.charAt(i[0]) == ']') {
                        i[0]++;
                        return l;
                    }
                    while (true) {
                        l.add(value(s, i));
                        ws(s, i);
                        if (s.charAt(i[0]++) == ']') {
                            return l;
                        }
                    }
                }
                case '"' -> {
                    i[0]++;
                    StringBuilder b = new StringBuilder();
                    while (true) {
                        char ch = s.charAt(i[0]++);
                        if (ch == '"') {
                            return b.toString();
                        }
                        if (ch == '\\') {
                            char e = s.charAt(i[0]++);
                            switch (e) {
                                case 'n' -> b.append('\n');
                                case 't' -> b.append('\t');
                                case 'r' -> b.append('\r');
                                case 'b' -> b.append('\b');
                                case 'f' -> b.append('\f');
                                case 'u' -> {
                                    b.append((char) Integer.parseInt(s.substring(i[0], i[0] + 4), 16));
                                    i[0] += 4;
                                }
                                default -> b.append(e);
                            }
                        } else {
                            b.append(ch);
                        }
                    }
                }
                default -> {
                    int start = i[0];
                    while (i[0] < s.length() && ",}] \n\r\t".indexOf(s.charAt(i[0])) < 0) {
                        i[0]++;
                    }
                    String tok = s.substring(start, i[0]);
                    return switch (tok) {
                        case "true" -> Boolean.TRUE;
                        case "false" -> Boolean.FALSE;
                        case "null" -> null;
                        default -> tok.contains(".") || tok.contains("e") || tok.contains("E") ? (Object) Double.parseDouble(tok) : (Object) Long.parseLong(tok);
                    };
                }
            }
        }
    }
}
