import java.lang.instrument.ClassFileTransformer;
import java.lang.instrument.Instrumentation;
import java.security.ProtectionDomain;
import java.util.Set;

import org.objectweb.asm.ClassReader;
import org.objectweb.asm.ClassVisitor;
import org.objectweb.asm.ClassWriter;
import org.objectweb.asm.MethodVisitor;
import org.objectweb.asm.Opcodes;

/**
 * Loaded by tcd into debug sessions only (-javaagent). It removes start-up work that costs nothing
 * natively but becomes very slow under a debugger, where every CUDA driver call is a debugger
 * round trip. It does not touch kernels, transfers or results.
 *
 * CUDACommandQueue.warmUpIssuePath: 256 x (H2D + D2H) throwaway copies, each with its own events,
 * per queue, only to pre-link FFM downcall handles for profiling accuracy. That is about 5,000 driver
 * calls, or about 20 s under cuda-gdb.
 *
 * Needs ASM, which TornadoVM ships as module org.objectweb.asm. The ASM version is not pinned.
 */
public final class TcdAgent {

    private static final String QUEUE = "uk/ac/manchester/tornado/drivers/cuda/CUDACommandQueue";
    private static final Set<String> SKIP = Set.of("warmUpIssuePath");

    public static void premain(String args, Instrumentation inst) {
        if (args != null && args.contains("ptracer")) {
            allowDebuggerAttach();
        }
        if (args != null && !args.contains("fast")) {
            return;
        }
        inst.addTransformer(new ClassFileTransformer() {
            @Override
            public byte[] transform(Module module, ClassLoader loader, String name, Class<?> redefined, ProtectionDomain pd, byte[] bytes) {
                if (!QUEUE.equals(name)) {
                    return null;
                }
                try {
                    return stubVoidMethods(bytes);
                } catch (Throwable t) {
                    System.err.println("tcd-agent: left " + name + " unchanged: " + t);
                    return null;
                }
            }
        });
    }

    /**
     * For `tcd launch`/`tcd attach`: with Yama ptrace_scope=1 a debugger may only attach to its own
     * children, unless the target opts in with prctl(PR_SET_PTRACER, PR_SET_PTRACER_ANY). This makes that
     * call. It goes through reflection over the FFM API, which is preview in JDK 21 (TornadoVM runs with
     * --enable-preview) and final from JDK 22, so this class compiles and loads on both.
     */
    static void allowDebuggerAttach() {
        try {
            Class<?> linkerC = Class.forName("java.lang.foreign.Linker");
            Class<?> fdC = Class.forName("java.lang.foreign.FunctionDescriptor");
            Class<?> layoutC = Class.forName("java.lang.foreign.MemoryLayout");
            Class<?> valueLayoutC = Class.forName("java.lang.foreign.ValueLayout");
            Class<?> segmentC = Class.forName("java.lang.foreign.MemorySegment");
            Class<?> lookupC = Class.forName("java.lang.foreign.SymbolLookup");
            Class<?> optionC = Class.forName("java.lang.foreign.Linker$Option");
            Object linker = linkerC.getMethod("nativeLinker").invoke(null);
            Object lookup = linkerC.getMethod("defaultLookup").invoke(linker);
            Object prctl = ((java.util.Optional<?>) lookupC.getMethod("find", String.class).invoke(lookup, "prctl")).orElseThrow();
            Object jInt = valueLayoutC.getField("JAVA_INT").get(null);
            Object jLong = valueLayoutC.getField("JAVA_LONG").get(null);
            Object args = java.lang.reflect.Array.newInstance(layoutC, 2);
            java.lang.reflect.Array.set(args, 0, jInt);
            java.lang.reflect.Array.set(args, 1, jLong);
            Object fd = fdC.getMethod("of", layoutC, args.getClass()).invoke(null, jInt, args);
            Object noOptions = java.lang.reflect.Array.newInstance(optionC, 0);
            java.lang.invoke.MethodHandle mh = (java.lang.invoke.MethodHandle) linkerC.getMethod("downcallHandle", segmentC, fdC, noOptions.getClass())
                    .invoke(linker, prctl, fd, noOptions);
            final int PR_SET_PTRACER = 0x59616d61;
            final long PR_SET_PTRACER_ANY = -1L;
            int rc = (int) mh.invokeWithArguments(PR_SET_PTRACER, PR_SET_PTRACER_ANY);
            System.err.println("tcd-agent: debugger attach " + (rc == 0 ? "enabled (pid " + ProcessHandle.current().pid() + ")" : "not enabled, prctl returned " + rc));
        } catch (Throwable t) {
            Throwable c = t instanceof java.lang.reflect.InvocationTargetException && t.getCause() != null ? t.getCause() : t;
            System.err.println("tcd-agent: could not enable debugger attach: " + c);
        }
    }

    static byte[] stubVoidMethods(byte[] bytes) {
        ClassReader reader = new ClassReader(bytes);
        ClassWriter writer = new ClassWriter(reader, ClassWriter.COMPUTE_MAXS);
        reader.accept(new ClassVisitor(Opcodes.ASM9, writer) {
            @Override
            public MethodVisitor visitMethod(int access, String name, String desc, String sig, String[] exceptions) {
                MethodVisitor mv = super.visitMethod(access, name, desc, sig, exceptions);
                if (SKIP.contains(name) && desc.endsWith(")V")) {
                    mv.visitCode();
                    mv.visitInsn(Opcodes.RETURN);
                    mv.visitMaxs(0, 0);
                    mv.visitEnd();
                    return null; // drop the original body
                }
                return mv;
            }
        }, 0);
        return writer.toByteArray();
    }
}
