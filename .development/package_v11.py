"""Make a portable v11 archive from the independently verified v10 archive."""
from pathlib import Path
import hashlib
import json
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def package():
    manifest = json.loads((ROOT/'artifacts/bge-m3-v11/manifest.json').read_text(encoding='utf-8'))
    assert sha256(ROOT/manifest['answer_file']) == manifest['answer_sha256']
    assert sha256(ROOT/manifest['ranker']) == manifest['ranker_sha256']
    assert sha256(ROOT/manifest['reference']) == manifest['reference_sha256']
    notebook = json.loads((ROOT/'solution.ipynb').read_text(encoding='utf-8'))
    config = json.loads((ROOT/'config/solution_v11.json').read_text(encoding='utf-8'))
    assert config['quality_v11']['answer_file'] == 'answer.csv'
    assert config['quality_v11']['answer_sha256'] == manifest['answer_sha256']
    portable = dict(manifest, answer_file='answer.csv')
    directory = ROOT/'deliverables/v11'
    directory.mkdir(parents=True, exist_ok=True)
    (directory/'solution.ipynb').write_text(json.dumps(notebook, ensure_ascii=False, indent=1),
                                         encoding='utf-8')
    (directory/'answer.csv').write_bytes((ROOT/manifest['answer_file']).read_bytes())
    (directory/'README.md').write_text('''# Avito v11

Положите исходные train.parquet, benchmark_queries.parquet и
benchmark_items.parquet рядом с solution.ipynb. Установите requirements.txt и
выполните Restart Kernel → Run All. Notebook создаст answer.csv с тем же
SHA-256, что и отправленный ответ. Все веса и справочники включены в архив;
внешние API не вызываются.

Обучение, проверка качества и ограничения описаны в docs/EXPERIMENTS_V11.md.
''', encoding='utf-8')
    replacements = {
        'solution.ipynb': (directory/'solution.ipynb').read_bytes(),
        'answer.csv': (directory/'answer.csv').read_bytes(),
        'README.md': (directory/'README.md').read_bytes(),
        'config/solution_v11.json': (ROOT/'config/solution_v11.json').read_bytes(),
        'artifacts/bge-m3-v11/manifest.json': json.dumps(portable, indent=2).encode('utf-8'),
        manifest['ranker']: (ROOT/manifest['ranker']).read_bytes(),
        manifest['reference']: (ROOT/manifest['reference']).read_bytes(),
    }
    for name in ['rank_features_v11.py', 'quality_model_v11.py',
                 'bge_reference_v11.py', 'quality_v11.py',
                 'build_v11_notebook.py', 'validate_v11.py',
                 'finalize_v11.py', 'polish_solution_notebook.py',
                 'run_validation_notebook.py']:
        path = ROOT/'.development'/name
        replacements[path.relative_to(ROOT).as_posix()] = path.read_bytes()
    docs = ROOT/'docs/EXPERIMENTS_V11.md'
    replacements[docs.relative_to(ROOT).as_posix()] = docs.read_bytes()
    overview = ROOT/'docs/SOLUTION.md'
    replacements[overview.relative_to(ROOT).as_posix()] = overview.read_bytes()
    archive = ROOT/'deliverables/solution_v11.zip'
    with zipfile.ZipFile(ROOT/'deliverables/avito_v10_solution.zip') as source:
        with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED,
                             compresslevel=1, allowZip64=True) as target:
            for info in source.infolist():
                if info.filename not in replacements and not info.filename.lower().endswith('.ipynb'):
                    with source.open(info) as input_stream, \
                            target.open(info.filename, 'w', force_zip64=True) as output_stream:
                        shutil.copyfileobj(input_stream, output_stream, 2**20)
            for name, data in replacements.items():
                target.writestr(name, data)
    print('Portable v11 archive', archive, 'bytes', archive.stat().st_size)


if __name__ == '__main__':
    package()
