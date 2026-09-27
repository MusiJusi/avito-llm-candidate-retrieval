"""Execute the bounded validation notebook and retain authentic outputs."""
import contextlib
import io
import json
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(encoding='utf-8')
os.chdir(ROOT)
path = ROOT / 'Avito_validation_v5.ipynb'
notebook = json.loads(path.read_text(encoding='utf-8'))
namespace = {'__name__': '__main__'}
execution = 0
for cell in notebook['cells']:
    if cell['cell_type'] != 'code':
        continue
    execution += 1
    stream = io.StringIO()
    cell['execution_count'] = execution
    try:
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            exec(compile(''.join(cell['source']), f'<validation cell {execution}>', 'exec'), namespace)
    except Exception as error:
        cell['outputs'] = [{'output_type': 'stream', 'name': 'stdout', 'text': stream.getvalue().splitlines(True)},
                           {'output_type': 'error', 'ename': type(error).__name__, 'evalue': str(error),
                            'traceback': traceback.format_exc().splitlines()}]
        path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding='utf-8')
        print(stream.getvalue(), flush=True)
        raise
    cell['outputs'] = [{'output_type': 'stream', 'name': 'stdout', 'text': stream.getvalue().splitlines(True)}] if stream.getvalue() else []
    print(f'Validation cell {execution} finished', flush=True)
    print(stream.getvalue(), flush=True)
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding='utf-8')
print('Validation preparation completed; no model training or answer.csv writes.', flush=True)
