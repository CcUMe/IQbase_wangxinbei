"""Termination keeps original failures and always attempts distributed cleanup."""
import os
from pathlib import Path
import signal
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.process_lifecycle import managed_entrypoint, _Lifecycle, _parent_death_signal


class ProcessLifecycleTests(unittest.TestCase):
    def test_normal_return_and_error_both_destroy_without_barrier(self):
        for failure in [False, True]:
            with self.subTest(failure=failure), patch('utils.process_lifecycle._parent_death_signal'), \
                 patch('torch.distributed.is_initialized', return_value=True), \
                 patch('torch.distributed.destroy_process_group') as destroy, \
                 patch('torch.distributed.barrier') as barrier:
                @managed_entrypoint('test')
                def entry():
                    if failure:
                        raise ValueError('original failure')
                    return 17
                if failure:
                    with self.assertRaisesRegex(ValueError, 'original failure'):
                        entry()
                else:
                    self.assertEqual(entry(), 17)
                destroy.assert_called_once()
                barrier.assert_not_called()

    def test_sigint_and_sigterm_restore_handlers_and_exit_with_signal_code(self):
        previous = {s: signal.getsignal(s) for s in [signal.SIGINT, signal.SIGTERM, signal.SIGHUP]}
        for signum in [signal.SIGINT, signal.SIGTERM]:
            with self.subTest(signum=signum), patch('utils.process_lifecycle._parent_death_signal'), \
                 patch('torch.distributed.is_initialized', return_value=True), \
                 patch('torch.distributed.destroy_process_group') as destroy, \
                 patch('utils.process_lifecycle._stderr'):
                @managed_entrypoint('test')
                def entry():
                    signal.getsignal(signum)(signum, None)
                with self.assertRaises(SystemExit) as caught:
                    entry()
                self.assertEqual(caught.exception.code, 128 + signum)
                destroy.assert_called_once()
                for key, handler in previous.items():
                    self.assertEqual(signal.getsignal(key), handler)

    def test_teardown_error_does_not_replace_original_failure(self):
        with patch('utils.process_lifecycle._parent_death_signal'), \
             patch('torch.distributed.is_initialized', return_value=True), \
             patch('torch.distributed.destroy_process_group', side_effect=RuntimeError('cleanup failure')), \
             patch('utils.process_lifecycle.threading.Timer') as timer, \
             patch('utils.process_lifecycle._stderr'):
            @managed_entrypoint('test')
            def entry():
                raise ValueError('original failure')
            with self.assertRaisesRegex(ValueError, 'original failure'):
                entry()
            timer.return_value.start.assert_called_once()
            timer.return_value.cancel.assert_not_called()

    def test_second_signal_is_local_force_exit(self):
        lifecycle = _Lifecycle('test', 10)
        lifecycle.stopping = True
        with patch('utils.process_lifecycle.os._exit', side_effect=SystemExit) as force:
            with self.assertRaises(SystemExit):
                lifecycle.request_stop(signal.SIGINT, None)
            force.assert_called_once_with(130)

    def test_non_worker_does_not_install_linux_parent_notification(self):
        with patch.dict(os.environ, {}, clear=True), patch('utils.process_lifecycle.ctypes.CDLL') as libc:
            self.assertIsNone(_parent_death_signal())
            libc.assert_not_called()


if __name__ == '__main__':
    unittest.main(verbosity=2)
