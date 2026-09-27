"""Package the main solution directly, without inheriting old release archives.

Only fitted inference dependencies, document/query vectors, reports and source
recipes are included. Large candidate pools, OOF matrices and raw data remain
local. Training can recreate those caches from the provided Parquet files.
"""
from pathlib import Path
import json
import zipfile
from validate_current import validate

ROOT = Path(__file__).resolve().parents[1]

def package():
    validate()
    manifest = json.loads((ROOT/'artifacts/query-encoder-candidate/manifest.json').read_text())
    paths = [ROOT/name for name in ['Avito.ipynb', 'answer.csv', 'README.md', 'requirements.txt', 'CHANGELOG.md']]
    for folder in ['docs', 'experiments', '.development', 'archive/notebooks', 'archive/documentation']:
        paths += [p for p in (ROOT/folder).rglob('*') if p.suffix in {'.py', '.md', '.ipynb'}]
    for folder in ['models/multilingual-e5-small', 'models/retrieval-priors']:
        paths += list((ROOT/folder).rglob('*'))
    baseline = ROOT/'artifacts/ranking-v5'
    paths += [p for p in baseline.iterdir() if p.suffix in {'.json', '.csv'} or
        p.name.startswith(('e5_items_', 'e5_queries_', 'final_ranker_', 'mixed_wide_', 'mixed_mined_'))]
    paths += [ROOT/'artifacts/ranking-v1'/name for name in
        ['final_ranker_c0442eeba2d244f2.joblib', 'lgb_rank_expanded_200_c0442eeba2d244f2.joblib']]
    paths += list((ROOT/manifest['encoder']).glob('*'))
    paths += [ROOT/manifest['query_vectors'], ROOT/manifest['final_ranker']]
    for folder in ['query-encoder-candidate', 'query-encoder-rank', 'query-encoder-pilot',
                   'cold-sensitivity-v5', 'cross-encoder-pilot', 'query-micro-blend', 'service-geography-v6']:
        paths += [p for p in (ROOT/'artifacts'/folder).glob('*') if p.suffix in {'.json', '.csv'}]
    paths += list((ROOT/'artifacts').glob('current*.json'))
    paths = sorted({p for p in paths if p.is_file() and not p.name.endswith('.tmp')})
    output = ROOT/'deliverables/avito_solution.zip'
    output.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for path in paths:
            assert path.resolve().is_relative_to(ROOT)
            archive.write(path, path.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
        assert archive.read('answer.csv') == (ROOT/'answer.csv').read_bytes()
        assert archive.read('Avito.ipynb') == (ROOT/'Avito.ipynb').read_bytes()
    print('Packaged', output.name, 'files', len(paths), 'MiB', round(output.stat().st_size/2**20, 1), flush=True)
    return output

if __name__ == '__main__':
    package()
