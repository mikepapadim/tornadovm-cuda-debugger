import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/**
 * Use case 2: the grid is rounded up to a multiple of the block size, but the
 * kernel has no bounds check, so the tail threads write past the end of y.
 *
 *   java ... Saxpy          # buggy kernel: saxpyBuggy
 *   java ... Saxpy fixed    # fixed kernel: saxpyFixed
 */
public class Saxpy {

    public static void saxpyBuggy(KernelContext ctx, float alpha, FloatArray x, FloatArray y) {
        int i = ctx.globalIdx;
        // BUG: no "if (i < y.getSize())" guard; the grid is larger than the array.
        y.set(i, alpha * x.get(i) + y.get(i));
    }

    public static void saxpyFixed(KernelContext ctx, float alpha, FloatArray x, FloatArray y) {
        int i = ctx.globalIdx;
        if (i < y.getSize()) {
            y.set(i, alpha * x.get(i) + y.get(i));
        }
    }

    public static void main(String[] args) throws Exception {
        boolean fixed = args.length > 0 && args[0].equals("fixed");
        int n = 1_000_000;
        int block = 256;
        int global = (n + block - 1) / block * block; // 1,000,192 threads for 1,000,000 elements
        FloatArray x = new FloatArray(n);
        FloatArray y = new FloatArray(n);
        for (int i = 0; i < n; i++) {
            x.set(i, i);
            y.set(i, 1);
        }
        KernelContext ctx = new KernelContext();
        WorkerGrid grid = new WorkerGrid1D(global);
        grid.setLocalWork(block, 1, 1);
        GridScheduler scheduler = new GridScheduler("blas.saxpy", grid);
        TaskGraph tg = new TaskGraph("blas").transferToDevice(DataTransferMode.FIRST_EXECUTION, x, y);
        if (fixed) {
            tg.task("saxpy", Saxpy::saxpyFixed, ctx, 2.0f, x, y);
        } else {
            tg.task("saxpy", Saxpy::saxpyBuggy, ctx, 2.0f, x, y);
        }
        tg.transferToHost(DataTransferMode.EVERY_EXECUTION, y);
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(tg.snapshot())) {
            plan.withGridScheduler(scheduler).execute();
        }
        System.out.printf("y[0]=%.1f y[n-1]=%.1f -> %s%n", y.get(0), y.get(n - 1), y.get(n - 1) == 2.0f * (n - 1) + 1 ? "PASS" : "FAIL");
    }
}
