"""Mirror console text to train.log, keeping tqdm redraws out of the saved text."""
import atexit
import re
import sys
import threading
from pathlib import Path


ANSI = re.compile(r'\x1b\[[0-?]*[ -/]*[@-~]')


class LogStream:
    def __init__(self, console, file, lock):
        self.console, self.file, self.lock = console, file, lock
        self.pending = ''

    def write(self, text):
        self.console.write(text)
        with self.lock:
            for fragment in re.split(r'([\r\n])', ANSI.sub('', text)):
                if fragment == '\r':
                    self.pending = ''
                elif fragment == '\n':
                    self.file.write(self.pending + '\n')
                    self.pending = ''
                else:
                    self.pending += fragment
        return len(text)

    def log_snapshot(self, text):
        with self.lock:
            self.file.write(ANSI.sub('', text).replace('\r', '') + '\n')
            self.file.flush()

    def flush(self):
        self.console.flush()
        with self.lock:
            self.file.flush()

    def isatty(self):
        return self.console.isatty()

    def fileno(self):
        return self.console.fileno()

    @property
    def encoding(self):
        return self.console.encoding


class ReadableRunLog:
    def __init__(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / 'train.log'
        self.migrated = None
        if path.exists():
            with path.open(encoding='utf-8') as old:
                first = old.readline().lstrip()
            if first.startswith('{"event"'):
                legacy = directory / 'train_legacy.jsonl'
                index = 1
                while legacy.exists():
                    legacy = directory / f'train_legacy_{index}.jsonl'
                    index += 1
                path.replace(legacy)
                self.migrated = legacy
        self.file = path.open('a', encoding='utf-8', buffering=1)
        self.original = (sys.stdout, sys.stderr)
        lock = threading.RLock()
        self.streams = (LogStream(sys.stdout, self.file, lock), LogStream(sys.stderr, self.file, lock))
        sys.stdout, sys.stderr = self.streams
        self.closed = False
        atexit.register(self.close)
        if self.migrated:
            print(f'原 JSON 日志已保留为 {self.migrated.name}；train.log 现在保存可读文本。', flush=True)

    def close(self):
        if not self.closed:
            for stream in self.streams:
                stream.flush()
                if stream.pending.strip():
                    self.file.write(stream.pending + '\n')
            sys.stdout, sys.stderr = self.original
            self.file.close()
            self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
