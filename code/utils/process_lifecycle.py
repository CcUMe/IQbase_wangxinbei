"""CLI signal handling and bounded distributed teardown, without final barriers."""
import ctypes
from functools import wraps
import os
import select
import signal
import sys
import threading


class TerminationRequested(BaseException):
    def __init__(self, signum):
        self.signum = signum


def _parent_death_signal():
    """Linux torchrun workers must not outlive an abruptly killed launcher."""
    if not sys.platform.startswith('linux') or 'LOCAL_RANK' not in os.environ:
        return None
    parent = os.getppid()
    if parent == 1:
        raise TerminationRequested(signal.SIGTERM)
    libc = ctypes.CDLL(None, use_errno=True)
    previous = ctypes.c_int()
    if libc.prctl(2, ctypes.byref(previous), 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), 'PR_GET_PDEATHSIG failed')
    # The agent is already gone, so no coordinator remains for graceful exit.
    # SIGKILL also works when a worker is blocked inside a C++ collective and
    # cannot run a Python SIGTERM handler. Normal stops still use SIGINT/TERM.
    if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), 'PR_SET_PDEATHSIG failed')
    # Close the race between reading PPID and arming the kernel notification.
    if os.getppid() != parent:
        raise TerminationRequested(signal.SIGTERM)
    return lambda: libc.prctl(1, previous.value, 0, 0, 0)


def _stderr(message):
    try:
        os.write(2, (message + '\n').encode())
    except OSError:
        pass  # The terminal itself may have been closed.


class _Lifecycle:
    def __init__(self, label, timeout):
        self.label, self.timeout = label, timeout
        self.timer = None
        self.stopping = False
        self.exit_code = 1
        self.timer_lock = threading.RLock()

    def arm_timeout(self, exit_code):
        with self.timer_lock:
            self.exit_code = exit_code
            if self.timer is None:
                self.timer = threading.Timer(self.timeout, self.force_exit)
                self.timer.daemon = True
                self.timer.start()

    def force_exit(self):
        _stderr(f'{self.label}退出清理超时，终止本进程，防止分布式后台线程残留。')
        os._exit(self.exit_code)

    def request_stop(self, signum, _frame):
        if self.stopping:
            os._exit(128 + signum)
        self.stopping = True
        # Start before stack unwinding: a CUDA destructor may itself block.
        self.arm_timeout(128 + signum)
        raise TerminationRequested(signum)

    def close_distributed(self, failed):
        import torch.distributed as dist
        if not dist.is_available() or not dist.is_initialized():
            return True
        self.arm_timeout(self.exit_code if self.stopping else 1)
        try:
            # Never add a barrier here: a peer may already have exited.
            dist.destroy_process_group()
        except Exception as error:
            _stderr(f'{self.label}通信组清理异常：{type(error).__name__}: {error}')
            if not failed:
                raise
            return False
        return True


class _SignalDeadline:
    """Observe the C-level signal pipe even if Python's main handler is delayed."""
    def __init__(self, lifecycle):
        self.lifecycle = lifecycle
        self.read_fd, self.write_fd = os.pipe2(os.O_NONBLOCK | os.O_CLOEXEC)
        self.previous_fd = signal.set_wakeup_fd(self.write_fd, warn_on_full_buffer=False)
        self.done = threading.Event()
        self.thread = threading.Thread(target=self.watch, daemon=True, name='signal-shutdown-deadline')
        self.thread.start()

    def watch(self):
        while not self.done.is_set():
            if not select.select([self.read_fd], [], [], .2)[0]:
                continue
            try:
                signals = os.read(self.read_fd, 4096)
            except BlockingIOError:
                continue
            for signum in signals:
                if signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
                    self.lifecycle.arm_timeout(128 + signum)

    def close(self):
        signal.set_wakeup_fd(self.previous_fd)
        self.done.set()
        try:
            os.write(self.write_fd, b'\x00')
        except BlockingIOError:
            pass
        self.thread.join(timeout=1)
        os.close(self.read_fd)
        os.close(self.write_fd)


def managed_entrypoint(label, shutdown_timeout=10.0):
    """Apply to CLI main; normal exit, exceptions and termination all clean up."""
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            lifecycle = _Lifecycle(label, shutdown_timeout)
            previous = {}
            restore_parent = None
            deadline = None
            try:
                if threading.current_thread() is threading.main_thread():
                    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
                        previous[signum] = signal.getsignal(signum)
                        signal.signal(signum, lifecycle.request_stop)
                    if sys.platform.startswith('linux'):
                        deadline = _SignalDeadline(lifecycle)
                    restore_parent = _parent_death_signal()
                return function(*args, **kwargs)
            except (TerminationRequested, KeyboardInterrupt) as error:
                signum = error.signum if isinstance(error, TerminationRequested) else signal.SIGINT
                lifecycle.stopping = True
                lifecycle.arm_timeout(128 + signum)
                rank = os.environ.get('RANK', '0')
                _stderr(f'{label}停止 | rank {rank} | {signal.Signals(signum).name} | 清理通信组；未额外保存中断检查点。')
                raise SystemExit(128 + signum) from None
            finally:
                failed = sys.exc_info()[0] is not None
                cleaned = False
                try:
                    cleaned = lifecycle.close_distributed(failed)
                finally:
                    if deadline is not None:
                        deadline.close()
                    if cleaned and lifecycle.timer is not None:
                        lifecycle.timer.cancel()
                    if restore_parent is not None:
                        restore_parent()
                    for signum, handler in previous.items():
                        signal.signal(signum, handler)
        return wrapped
    return decorate
