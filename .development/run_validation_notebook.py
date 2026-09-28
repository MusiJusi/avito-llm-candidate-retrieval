"""Execute a standalone notebook and retain authentic cell outputs."""
import argparse
import contextlib
import hashlib
import io
import json
import os
import sys
import time
import traceback
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(encoding='utf-8')
os.chdir(ROOT)
parser = argparse.ArgumentParser()
parser.add_argument('notebook', nargs='?', default='solution.ipynb')
arguments = parser.parse_args()
path = ROOT / arguments.notebook
if not path.resolve().is_relative_to(ROOT) or path.suffix != '.ipynb':
    raise ValueError('Notebook must be an .ipynb file within the project')
notebook = json.loads(path.read_text(encoding='utf-8'))
network_disabled = os.environ.get('AVITO_DISABLE_NETWORK') == '1'
if network_disabled:
    import socket
    def deny_network(*args, **kwargs):
        raise RuntimeError('Network connections are disabled during reproduction')
    socket.socket.connect = deny_network
    socket.socket.connect_ex = deny_network
    socket.create_connection = deny_network
# Match a Jupyter kernel: pickle/joblib must resolve notebook-defined classes
# through the actual __main__ module, rather than an unrelated exec dictionary.
kernel_module = types.ModuleType('__main__')
sys.modules['__main__'] = kernel_module
namespace = kernel_module.__dict__
execution = 0
started = time.perf_counter()
for cell in notebook['cells']:
    if cell['cell_type'] != 'code':
        continue
    execution += 1
    stream = io.StringIO()
    cell['execution_count'] = execution
    try:
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            exec(compile(''.join(cell['source']), f'<notebook cell {execution}>', 'exec'), namespace)
    except Exception as error:
        cell['outputs'] = [{'output_type': 'stream', 'name': 'stdout', 'text': stream.getvalue().splitlines(True)},
                           {'output_type': 'error', 'ename': type(error).__name__, 'evalue': str(error),
                            'traceback': traceback.format_exc().splitlines()}]
        path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding='utf-8')
        print(stream.getvalue(), flush=True)
        raise
    cell['outputs'] = [{'output_type': 'stream', 'name': 'stdout', 'text': stream.getvalue().splitlines(True)}] if stream.getvalue() else []
    print(f'Notebook cell {execution} finished', flush=True)
    print(stream.getvalue(), flush=True)
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding='utf-8')
report = {'notebook': arguments.notebook, 'code_cells_executed': execution,
          'code_sha256': hashlib.sha256('\n'.join(''.join(cell['source']) for cell in notebook['cells']
              if cell['cell_type'] == 'code').encode('utf-8')).hexdigest(),
          'device': namespace.get('DEVICE', 'not_used'),
          'network_connections_disabled': network_disabled,
          'elapsed_seconds': round(time.perf_counter() - started, 2)}
if (ROOT / 'answer.csv').exists():
    report['answer_sha256'] = hashlib.sha256((ROOT / 'answer.csv').read_bytes()).hexdigest()
(ROOT / 'artifacts' / 'notebook_execution.json').write_text(
    json.dumps(report, indent=2), encoding='utf-8')
print('Notebook execution completed:', json.dumps(report), flush=True)
