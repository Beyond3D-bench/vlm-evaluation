#!/usr/bin/env python3
import datetime as _datetime
import faulthandler
import os
import runpy
import sys


def _stack_dump_interval() -> int:
    raw = os.environ.get("OOS_STACK_DUMP_SEC", "0")
    try:
        return int(raw)
    except ValueError:
        print(f"Invalid OOS_STACK_DUMP_SEC={raw!r}; disabling stack dumps", file=sys.stderr, flush=True)
        return 0


faulthandler.enable(all_threads=True)
interval = _stack_dump_interval()
if interval > 0:
    faulthandler.dump_traceback_later(interval, repeat=True, file=sys.stderr)

print(
    f"Python wrapper: entering lmms_eval at {_datetime.datetime.now().isoformat(timespec='seconds')}",
    flush=True,
)

sys.argv = ["lmms_eval"] + sys.argv[1:]
runpy.run_module("lmms_eval", run_name="__main__", alter_sys=True)
