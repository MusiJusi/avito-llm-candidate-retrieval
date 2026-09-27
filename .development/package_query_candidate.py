"""Add the learned query candidate to the already verified local solution bundle."""
from pathlib import Path
import json
import zipfile
ROOT=Path(__file__).resolve().parents[1]
CACHE=ROOT/'artifacts/query-encoder-candidate'
manifest=json.loads((CACHE/'manifest.json').read_text())
additional=[Path(__file__).resolve(),ROOT/'deliverables/query_encoder_submission/answer.csv',
    ROOT/'Avito_query_encoder_candidate.ipynb',ROOT/manifest['answer_file'],
    ROOT/'README_query_encoder.md',ROOT/manifest['query_vectors']]
additional+=list((ROOT/manifest['encoder']).rglob('*'))
additional+=[p for p in CACHE.iterdir() if p.suffix in {'.json','.csv'}]
if manifest['final_ranker']:additional.append(ROOT/manifest['final_ranker'])
additional=sorted({p for p in additional if p.is_file()})
assert all(p.is_relative_to(ROOT) for p in additional)
output=ROOT/'deliverables/avito_query_encoder_solution.zip'
with zipfile.ZipFile(ROOT/'deliverables/avito_v6_experiments.zip') as previous:
    with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as archive:
        replacement_names={path.relative_to(ROOT).as_posix() for path in additional}
        for name in previous.namelist():
            if name not in replacement_names:archive.writestr(name,previous.read(name))
        for path in additional:archive.write(path,path.relative_to(ROOT).as_posix())
with zipfile.ZipFile(output) as archive:
    assert archive.testzip() is None
    assert archive.read(manifest['answer_file'])==(ROOT/manifest['answer_file']).read_bytes()
    assert archive.read('deliverables/query_encoder_submission/answer.csv')==(ROOT/manifest['answer_file']).read_bytes()
print('Packaged learned-query solution MiB:',round(output.stat().st_size/2**20,1),flush=True)
