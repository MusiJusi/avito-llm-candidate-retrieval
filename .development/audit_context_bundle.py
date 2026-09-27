"""Check that the final archive contains the actually reproduced notebook."""
from pathlib import Path
import ast
import hashlib
import json
import zipfile

ROOT=Path(__file__).resolve().parents[1]
manifest=json.loads((ROOT/'artifacts/context-v9/manifest.json').read_text())
proof=json.loads((ROOT/'artifacts/context-v9/reproduction.json').read_text())
with zipfile.ZipFile(ROOT/'deliverables/avito_v9_solution.zip') as archive:
    names=archive.namelist()
    assert len(names)==len(set(names))
    notebook=json.loads(archive.read('Avito.ipynb'))
    cells=[c for c in notebook['cells'] if c['cell_type']=='code']
    for cell in cells:
        ast.parse(''.join(cell['source']))
        assert not any(output.get('output_type')=='error' for output in cell['outputs'])
    assert [c['execution_count'] for c in cells]==list(range(1,len(cells)+1))
    digest=hashlib.sha256('\n'.join(''.join(c['source']) for c in cells).encode()).hexdigest()
    assert digest==proof['notebook_code_sha256']
    assert hashlib.sha256(archive.read('answer.csv')).hexdigest()==manifest['answer_sha256']
    assert json.loads(archive.read('artifacts/context-v9/reproduction.json'))==proof
    dependencies=['history_bank','query_evidence','context_ranker','final_ranker','query_vectors']
    for key in dependencies:
        if manifest.get(key):assert manifest[key] in names,key
    assert hashlib.sha256(archive.read('.development/context_signals_v9.py')).hexdigest()==manifest['context_module_sha256']
assert hashlib.sha256((ROOT/'answer.csv').read_bytes()).hexdigest()=='a256639b72ef3bd515bf5118c5e337aee23a34c9e0b261da580b1c852411a6b2'
result={'valid':True,'archive_notebook_actually_executed':True,'code_cells':len(cells),
        'notebook_code_sha256':digest,'answer_sha256':manifest['answer_sha256'],
        'primary_answer_unchanged':True}
(ROOT/'artifacts/context-v9/bundle_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2),flush=True)
