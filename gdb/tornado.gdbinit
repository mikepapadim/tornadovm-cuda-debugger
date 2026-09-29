# tornado.gdbinit - cuda-gdb settings for debugging a JVM running TornadoVM.
set pagination off
set confirm off
set breakpoint pending on
set print pretty on
# HotSpot uses these signals internally (safepoints, implicit null checks,
# thread suspension). Stopping on them makes the JVM undebuggable.
handle SIGSEGV SIGBUS SIGILL SIGFPE SIGQUIT SIGUSR1 SIGUSR2 SIGPIPE nostop noprint pass
# The CUDA driver reports expected API errors while TornadoVM probes devices.
set cuda api_failures ignore
