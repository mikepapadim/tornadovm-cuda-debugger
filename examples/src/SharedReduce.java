import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/**
 * Use case 1: a block-level sum reduction in shared memory that forgot a barrier.
 *
 * Each block of 256 threads sums 256 inputs into partial[groupIdx]. The buggy kernel
 * reads a neighbour's shared-memory slot before that neighbour has written it,
 * so the partial sums are wrong. Warps run out of step, so exactly how wrong varies from run to run.
 *
 *   java ... SharedReduce          # buggy kernel: reduceBuggy
 *   java ... SharedReduce fixed    # fixed kernel: reduceFixed
 */
public class SharedReduce {

    static final int BLOCK = 256;

    public static void reduceBuggy(KernelContext ctx, FloatArray input, FloatArray partial) {
        int gid = ctx.globalIdx;
        int lid = ctx.localIdx;
        float[] scratch = ctx.allocateFloatLocalArray(BLOCK);
        scratch[lid] = input.get(gid);
        // BUG: missing ctx.localBarrier() here and inside the loop.
        for (int stride = BLOCK / 2; stride > 0; stride /= 2) {
            if (lid < stride) {
                scratch[lid] += scratch[lid + stride];
            }
        }
        if (lid == 0) {
            partial.set(ctx.groupIdx, scratch[0]);
        }
    }

    public static void reduceFixed(KernelContext ctx, FloatArray input, FloatArray partial) {
        int gid = ctx.globalIdx;
        int lid = ctx.localIdx;
        float[] scratch = ctx.allocateFloatLocalArray(BLOCK);
        scratch[lid] = input.get(gid);
        ctx.localBarrier();
        for (int stride = BLOCK / 2; stride > 0; stride /= 2) {
            if (lid < stride) {
                scratch[lid] += scratch[lid + stride];
            }
            ctx.localBarrier();
        }
        if (lid == 0) {
            partial.set(ctx.groupIdx, scratch[0]);
        }
    }

    public static void main(String[] args) throws Exception {
        boolean fixed = args.length > 0 && args[0].equals("fixed");
        int n = 1024 * BLOCK;
        int groups = n / BLOCK;
        FloatArray input = new FloatArray(n);
        FloatArray partial = new FloatArray(groups);
        for (int i = 0; i < n; i++) {
            input.set(i, 1.0f);
        }

        KernelContext ctx = new KernelContext();
        WorkerGrid grid = new WorkerGrid1D(n);
        grid.setLocalWork(BLOCK, 1, 1);
        GridScheduler scheduler = new GridScheduler("reduce.sum", grid);

        TaskGraph tg = new TaskGraph("reduce").transferToDevice(DataTransferMode.FIRST_EXECUTION, input);
        if (fixed) {
            tg.task("sum", SharedReduce::reduceFixed, ctx, input, partial);
        } else {
            tg.task("sum", SharedReduce::reduceBuggy, ctx, input, partial);
        }
        tg.transferToHost(DataTransferMode.EVERY_EXECUTION, partial);

        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(tg.snapshot())) {
            plan.withGridScheduler(scheduler).execute();
        }

        int wrong = 0;
        for (int g = 0; g < groups; g++) {
            if (partial.get(g) != BLOCK) {
                if (wrong < 5) {
                    System.out.printf("partial[%d] = %.1f (expected %d)%n", g, partial.get(g), BLOCK);
                }
                wrong++;
            }
        }
        System.out.printf("%s: %d/%d blocks wrong -> %s%n", fixed ? "reduceFixed" : "reduceBuggy", wrong, groups, wrong == 0 ? "PASS" : "FAIL");
    }
}
