"""Promote the verified portable v11 notebook and answer to project root."""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def promote():
    manifest = json.loads((ROOT/'artifacts/bge-m3-v11/manifest.json').read_text(encoding='utf-8'))
    report = json.loads((ROOT/'artifacts/bge-m3-v11/reproduction.json').read_text(encoding='utf-8'))
    assert report['answer_bytes_equal'] and report['network_connections_disabled']
    answer = (ROOT/manifest['answer_file']).read_bytes()
    digest = hashlib.sha256(answer).hexdigest()
    assert digest == manifest['answer_sha256'] == report['answer_sha256']
    with zipfile.ZipFile(ROOT/'deliverables/avito_v11_solution.zip') as archive:
        assert archive.read('answer.csv') == answer
        notebook = archive.read('Avito.ipynb')
    parsed = json.loads(notebook)
    assert any("'answer_file': 'answer.csv'" in ''.join(cell['source'])
               for cell in parsed['cells'] if cell['cell_type'] == 'code')
    (ROOT/'answer.csv').write_bytes(answer)
    (ROOT/'Avito.ipynb').write_bytes(notebook)
    details = {
        'notebook': 'Avito.ipynb',
        'code_cells_executed': report['code_cells_executed'],
        'code_sha256': report['notebook_code_sha256'],
        'device': 'cpu',
        'network_connections_disabled': True,
        'elapsed_seconds': report['elapsed_seconds'],
        'answer_sha256': digest,
        'fresh_extraction': True,
        'answer_bytes_equal': True,
    }
    (ROOT/'artifacts/notebook_execution.json').write_text(
        json.dumps(details, indent=2), encoding='utf-8')
    assert hashlib.sha256((ROOT/'answer.csv').read_bytes()).hexdigest() == digest
    print('Promoted verified v11', digest)


if __name__ == '__main__':
    promote()
