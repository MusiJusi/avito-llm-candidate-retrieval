"""Serialize large CPU experiments to fit within laptop RAM.

This driver never submits to the platform. Development chooses each model;
control is reported after selection. Logs survive a disconnected notebook.
"""
from pathlib import Path
import ctypes
import json
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
CACHE=ROOT/'artifacts/quality-v10'


def wait_process(pid):
    kernel=ctypes.windll.kernel32
    kernel.OpenProcess.restype=ctypes.c_void_p
    handle=kernel.OpenProcess(0x00100000,False,pid)
    if handle:
        try:
            while kernel.WaitForSingleObject(ctypes.c_void_p(handle),10000)==258:
                pass
        finally:kernel.CloseHandle(ctypes.c_void_p(handle))


def run(name):
    path=ROOT/'.development'/name
    log=CACHE/(path.stem+'.log')
    print('Start',name,time.strftime('%Y-%m-%d %H:%M:%S'),flush=True)
    with log.open('a',encoding='utf-8') as output:
        result=subprocess.run([sys.executable,'-u',str(path)],cwd=ROOT,stdout=output,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError(f'{name} failed; see {log}')
    print('Finished',name,flush=True)


def main():
    # The first expanded-ranker comparison is already running when launched.
    if len(sys.argv)>1:wait_process(int(sys.argv[1]))
    assert (CACHE/'control.json').exists(),'Base comparison must finish first'
    if not (CACHE/'no_popularity_control.json').exists():run('no_popularity_v10.py')
    run('warm_history_v10.py')
    selection=json.loads((CACHE/'selection.json').read_text())
    if selection['winner']['variant']=='warm_mixed':
        warm_control=json.loads((ROOT/'artifacts/warm-history-v10/control.json').read_text())
        original=json.loads((CACHE/'control.json').read_text())
        (CACHE/'control.json').write_text(json.dumps(dict(original,
            before_warm_training=original,selected_recall50=warm_control['selected_recall50']),indent=2),encoding='utf-8')
        selection['recipes']['warm_mixed']={
            'columns':list(range(86)),'trees':500,'leaves':31,'min_child':80}
        (CACHE/'selection.json').write_text(json.dumps(selection,indent=2),encoding='utf-8')
    for name in ['warm_deeper_v10.py','check_category_shift_v10.py','field_ranker_v10.py','history_sensitivity_v10.py']:
        run(name)
    if len(sys.argv)>2:wait_process(int(sys.argv[2]))
    if (ROOT/'artifacts/bge-m3-v10/vectors.json').exists():
        run('bge_ranker_v10.py')
    else:
        (CACHE/'bge_unavailable.json').write_text(json.dumps({
            'included':False,'reason':'Model download/encoding did not complete; see bge logs.'},indent=2),encoding='utf-8')
    for name in ['export_quality_v10.py','validate_quality_v10.py',
                 'write_quality_report.py','package_quality_v10.py','verify_quality_v10.py',
                 'write_quality_report.py','package_quality_v10.py','audit_quality_bundle_v10.py']:
        run(name)
    print('QUALITY V10 COMPLETE',flush=True)


if __name__=='__main__':main()
