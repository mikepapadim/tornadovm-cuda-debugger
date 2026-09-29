///usr/bin/env jbang "$0" "$@" ; exit $?
//JAVA 21+
//FILES gdb/tornado.gdbinit=gdb/tornado.gdbinit
//FILES gdb/tornado_gdb.py=gdb/tornado_gdb.py
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
                  version    print version

                Options:
                  --break, -b K[:L]   break in device kernel K (a Java method name), optionally at
                                      line L of the generated CUDA C. Repeatable.
                  --at B:T            thread to inspect: 1-D "2:5" or 3-D "2,0,0:5,0,0". Repeatable.
                  --print, -p e1,e2   extra expressions to evaluate per thread
                  --hits N            batch: stop after N breakpoint hits (default 1)
                  --json              batch: machine-readable output
                  --timeout SECS      batch: kill the session after SECS seconds (default 600)
                  --tool NAME         memcheck: sanitizer tool(s) (memcheck, racecheck, initcheck, synccheck)
                  --verbose, -v       memcheck: print every sanitizer report, not only the first 3
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
                if (!keep) {
                    deleteTree(dumpAbs);
                }
            } catch (IOException e) {
                err.println("tcd: could not collect kernel dumps: " + e.getMessage());
            }
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

    // ------------------------------------------------------------------ batch

    static int batch(Options o) throws Exception {
        requireApp(o);
        if (o.breaks.isEmpty()) {
            err.println("tcd batch: give at least one --break KERNEL[:LINE]");
            return 2;
        }
        Session s = new Session(tornadoHome(o), o.keep);
        List<String> cmd = gdbBase(o, false);
        cmd.addAll(List.of("-batch", "-ex", "tcd-batch", "--args"));
        cmd.addAll(s.javaCommand(o, "-G -lineinfo"));

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
        spec.append("}");

        ProcessBuilder pb = new ProcessBuilder(cmd).redirectErrorStream(true);
        pb.environment().putAll(s.env());
        pb.environment().put("TCD_BATCH", spec.toString());
        if (!o.json) {
            err.println("tcd: running under cuda-gdb (session " + s.dir + ") ...");
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
        Files.writeString(s.dir.resolve("gdb.log"), log);
        s.close();
        if (report == null) {
            err.println("tcd: no report produced (cuda-gdb exit " + rc + "). Log: " + s.dir.resolve("gdb.log"));
            log.toString().lines().filter(l -> !l.startsWith("[New Thread") && !l.contains("exited]")).skip(Math.max(0, log.toString().lines().count() - 25)).forEach(err::println);
            return 1;
        }
        Object doc = Json.parse(report);
        if (o.json) {
            out.println(report);
        } else {
            printBatch(Json.path(doc, "batch"), s);
        }
        return 0;
    }

    @SuppressWarnings("unchecked")
    static void printBatch(Object b, Session s) {
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
            for (Object t : (List<Object>) Json.path(h, "threads")) {
                out.printf("%n-- block %s thread %s  (line %s)%n", coord(Json.path(t, "block")), coord(Json.path(t, "thread")), Json.path(t, "line"));
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
                if (locals != null) {
                    locals.entrySet().stream().filter(e -> !e.getKey().startsWith("_")).sorted(Map.Entry.comparingByKey(tcd::naturalOrder))
                            .forEach(e -> out.printf("   %-20s = %s%n", e.getKey(), e.getValue()));
                }
            }
        }
        out.println("\n(" + Json.path(b, "exit") + "; generated sources in " + s.dir.resolve("dumps") + ")");
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

    // ------------------------------------------------------------------ memcheck

    static final Pattern SANITIZER_FRAME = Pattern.compile("(\\w+)\\+0x[0-9a-fA-F]+ in \\S*tornado_kernel\\.cu:(\\d+)");

    static int memcheck(Options o) throws Exception {
        requireApp(o);
        Session s = new Session(tornadoHome(o), o.keep);
        Path san = cudaTool("compute-sanitizer", null).orElseThrow(() -> new IllegalStateException("compute-sanitizer not found"));
        List<String> tools = o.tools.isEmpty() ? List.of("memcheck") : o.tools;
        int worst = 0;
        for (String tool : tools) {
            List<String> cmd = new ArrayList<>(List.of(san.toString(), "--tool", tool, "--show-backtrace", "device"));
            // -lineinfo only: -G would serialize the kernel and can hide races.
            cmd.addAll(s.javaCommand(o, "-lineinfo"));
            err.println("tcd: compute-sanitizer --tool " + tool + " (session " + s.dir + ")");
            ProcessBuilder pb = new ProcessBuilder(cmd).redirectErrorStream(true);
            pb.environment().putAll(s.env());
            Process p = pb.start();
            SanitizerSummary summary = new SanitizerSummary(tool);
            try (BufferedReader r = new BufferedReader(new InputStreamReader(p.getInputStream(), StandardCharsets.UTF_8))) {
                for (String line; (line = r.readLine()) != null;) {
                    Matcher m = SANITIZER_FRAME.matcher(line);
                    String location = m.find() ? m.group(1) + ":" + m.group(2) : null;
                    String code = location == null ? null : generatedLine(s, m.group(1), Integer.parseInt(m.group(2)));
                    summary.accept(line, location, code);
                    if (o.verbose || !line.startsWith("=========") || summary.reported <= 3 || line.contains("SUMMARY")) {
                        out.println(line);
                        if (code != null) {
                            out.println("=========         >> " + location + "  " + code);
                        }
                    }
                }
            }
            worst = Math.max(worst, p.waitFor());
            summary.print(o.verbose);
        }
        s.close();
        err.println("tcd: generated kernel sources: " + s.dir.resolve("dumps"));
        return worst;
    }

    /**
     * Groups sanitizer reports by (error kind, kernel line). A missing bounds check on
     * a grid tail produces one report per thread; developers want one line per bug.
     */
    static final class SanitizerSummary {
        static final Pattern HEADER = Pattern.compile("^========= (Invalid .*|Error: .*|Race reported.*|Uninitialized .*|Barrier error.*|Program hit .*)");
        static final Pattern THREAD = Pattern.compile("by thread \\((\\d+),(\\d+),(\\d+)\\) in block \\((\\d+),(\\d+),(\\d+)\\)");
        final String tool;
        final Map<String, int[]> counts = new LinkedHashMap<>();
        final Map<String, String> firstThread = new LinkedHashMap<>();
        final Map<String, String> codeOf = new LinkedHashMap<>();
        String kind;
        String key;
        int reported;

        SanitizerSummary(String tool) {
            this.tool = tool;
        }

        void accept(String line, String location, String code) {
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
                    out.println("             first: " + firstThread.get(k) + "   -> tcd batch -b " + k.substring(k.lastIndexOf("@ ") + 2) + " --at "
                            + firstThread.get(k).replaceAll("block \\(([\\d,]+)\\) thread \\(([\\d,]+)\\)", "$1:$2"));
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
                    String spec = b.get("kernel") + (b.get("line") != null ? ":" + ((Number) b.get("line")).intValue() : "");
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
