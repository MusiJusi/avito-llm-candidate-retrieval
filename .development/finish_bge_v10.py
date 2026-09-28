"""Finish the bounded encoder pilot before freezing the final ensemble.

The earlier encoder process may have timed out while weights were downloading.
This coordinator waits for that process, resumes only when weights are verified,
and never lets two processes write the same embedding matrix.
"""
from pathlib import Path
import ctypes
import json
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]


def finish():
    cache=ROOT/'artifacts/quality-v10'
    path=cache/'bge_process.json'
    if not path.exists():return
    configuration=json.loads(path.read_text())
    if Path(configuration['workspace']).resolve()!=ROOT.resolve():return
    kernel=ctypes.windll.kernel32;kernel.OpenProcess.restype=ctypes.c_void_p
    handle=kernel.OpenProcess(0x00100000,False,configuration['encoder_pid'])
    if handle:
        try:
            while kernel.WaitForSingleObject(ctypes.c_void_p(handle),10000)==258:pass
        finally:kernel.CloseHandle(ctypes.c_void_p(handle))
    if (ROOT/'artifacts/bge-m3-v10/vectors.json').exists():return
    checkpoint=ROOT/'models/bge-m3/manifest.json';deadline=time.monotonic()+5400
    while not checkpoint.exists() and time.monotonic()<deadline:
        print('Waiting for verified BGE checkpoint; other comparisons are complete.',flush=True)
        time.sleep(30)
    if not checkpoint.exists():
        print('BGE checkpoint unavailable after bounded wait; keep proven improvements.',flush=True)
        return
    with (cache/'bge_encoding_resume.log').open('w',encoding='utf-8') as output:
        result=subprocess.run([sys.executable,'-u',str(ROOT/'.development/encode_bge_m3_v10.py')],
            cwd=ROOT,stdout=output,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError('BGE encoding failed; see bge_encoding_resume.log')


if __name__=='__main__':finish()
