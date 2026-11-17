"""Optional Python-only detached CLI startup; no shell or bash dependency."""
from datetime import datetime
import os
from pathlib import Path
import subprocess
import sys
import time
from zoneinfo import ZoneInfo


def launch_background(log_root):
    directory=Path(log_root);directory.mkdir(parents=True,exist_ok=True)
    stamp=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S_%f')
    log=directory/(stamp+'.log')
    arguments=[x for x in sys.argv[1:] if x!='--background']
    env={**os.environ,'OPENBLAS_NUM_THREADS':'4','OMP_NUM_THREADS':'4','PYTHONUNBUFFERED':'1'}
    with log.open('wb') as stream:
        child=subprocess.Popen([sys.executable,'-u',str(Path(sys.argv[0]).resolve()),*arguments],
            stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=env,start_new_session=True)
    log.with_suffix('.pid').write_text(str(child.pid)+'\n')
    time.sleep(2)
    code=child.poll()
    if code is not None and code!=0:
        print(log.read_text(),file=sys.stderr)
        raise SystemExit(code)
    print(f'PID={child.pid}\n启动日志: {log.resolve()}',flush=True)
