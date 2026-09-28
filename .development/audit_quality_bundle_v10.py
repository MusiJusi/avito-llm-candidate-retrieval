"""Check the final upload, executed notebook and portable dependency closure."""
from pathlib import Path
import ast
import hashlib
import json
import zipfile

ROOT=Path(__file__).resolve().parents[1]


def digest(data):return hashlib.sha256(data).hexdigest()


def audit():
    manifest=json.loads((ROOT/'artifacts/quality-v10/manifest.json').read_text())
    proof=json.loads((ROOT/'artifacts/quality-v10/reproduction.json').read_text())
    validation=json.loads((ROOT/'artifacts/quality-v10/answer_validation.json').read_text())
    archive_path=ROOT/'deliverables/avito_v10_solution.zip'
    with zipfile.ZipFile(archive_path) as archive:
        names=set(archive.namelist())
        assert not any(name.endswith('.parquet') for name in names)
        notebook=json.loads(archive.read('Avito.ipynb'))
        code=[cell for cell in notebook['cells'] if cell['cell_type']=='code']
        code_hash=digest('\n'.join(''.join(cell['source']) for cell in code).encode())
        assert code_hash==proof['notebook_code_sha256']
        assert len(code)==proof['code_cells_executed']
        assert [cell['execution_count'] for cell in code]==list(range(1,len(code)+1))
        assert not any(output.get('output_type')=='error' for cell in code for output in cell['outputs'])
        bindings={}
        for cell in code:
            text=''.join(cell['source'])
            if text.startswith(('FROZEN_QUERY_MANIFEST = ','QUALITY_V10_MANIFEST = ')):
                node=ast.parse(text).body[0]
                bindings[node.targets[0].id]=ast.literal_eval(node.value)
        assert bindings['QUALITY_V10_MANIFEST']['answer_file']=='answer.csv'
        assert bindings['FROZEN_QUERY_MANIFEST']['answer_file']=='answer.csv'
        assert digest(archive.read('answer.csv'))==manifest['answer_sha256']==proof['answer_sha256']
        dependencies=dict(manifest['module_sha256'])
        dependencies.update(manifest.get('cross_files_sha256',{}))
        for key in ['history_bank','contextual_vectors','ranker','field_ranker','field_teacher_vectors','bge_queries','bge_ranker','warm_aux_ranker']:
            if manifest.get(key):dependencies[manifest[key]]=manifest[key+'_sha256']
        for details in manifest.get('field_documents',{}).values():dependencies[details['path']]=details['sha256']
        if manifest.get('bge_kind'):
            details=manifest['bge_documents'];dependencies[details['path']]=details['sha256']
            for filename,expected in manifest['bge_checkpoint']['files_sha256'].items():
                dependencies['models/bge-m3/'+filename]=expected
        for filename,expected in dependencies.items():
            assert filename in names,filename
            hasher=hashlib.sha256()
            with archive.open(filename) as stream:
                for block in iter(lambda:stream.read(8*1024*1024),b''):hasher.update(block)
            assert hasher.hexdigest()==expected,filename
        assert digest(archive.read('artifacts/quality-v10/reproduction.json'))==digest((ROOT/'artifacts/quality-v10/reproduction.json').read_bytes())
    assert digest((ROOT/'answer.csv').read_bytes())=='a256639b72ef3bd515bf5118c5e337aee23a34c9e0b261da580b1c852411a6b2'
    assert digest((ROOT/'experiments/results/v9/answer.csv').read_bytes())=='a78cd853793bcba0087573333d830d439c5e94dbae9b3d23fc3eecc2c35fee82'
    report={'archive':archive_path.relative_to(ROOT).as_posix(),'archive_bytes':archive_path.stat().st_size,
        'answer_sha256':manifest['answer_sha256'],'all_new_dependencies_present_and_hash_match':True,
        'executed_notebook_code_matches_cpu_proof':True,'executed_cells':len(code),
        'v7_and_v9_answers_unchanged':True,'raw_data_excluded':True,'csv_validation':validation}
    (ROOT/'artifacts/quality-v10/bundle_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('Quality bundle verified',json.dumps(report),flush=True)


if __name__=='__main__':audit()
