"""Portable v10 archive; inherited v9 dependencies plus selected new weights."""
from pathlib import Path
import ast
import hashlib
import json
import zipfile

ROOT=Path(__file__).resolve().parents[1]


def package():
    manifest=json.loads((ROOT/'artifacts/quality-v10/manifest.json').read_text())
    notebook=json.loads((ROOT/'experiments/Avito_v10_candidate.ipynb').read_text(encoding='utf-8'))
    portable=dict(manifest,answer_file='answer.csv')
    notebook['cells'][0]['source']=['# Avito v10\n',
        '\nRestart Kernel → Run All создаёт answer.csv из локальных весов без внешних API.\n',
        'Описание, обучение и измерения: docs/EXPERIMENTS_V10.md.\n']
    for cell in notebook['cells']:
        if cell['cell_type']!='code':continue
        text=''.join(cell['source'])
        if text.startswith('FROZEN_QUERY_MANIFEST = '):
            node=ast.parse(text).body[0]
            frozen=ast.literal_eval(node.value)
            frozen.update(answer_file='answer.csv',answer_sha256=manifest['answer_sha256'])
            text='FROZEN_QUERY_MANIFEST = '+repr(frozen)+'\n'+text[text.index('\n')+1:]
        if text.startswith('QUALITY_V10_MANIFEST = '):text='QUALITY_V10_MANIFEST = '+repr(portable)+'\n'
        ast.parse(text);cell['source']=text.splitlines(True)
    directory=ROOT/'deliverables/v10';directory.mkdir(parents=True,exist_ok=True)
    def code_hash(nb):return hashlib.sha256('\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code').encode()).hexdigest()
    proof_path=ROOT/'artifacts/quality-v10/reproduction.json'
    existing=directory/'Avito.ipynb'
    if existing.exists() and proof_path.exists():
        executed=json.loads(existing.read_text(encoding='utf-8'))
        proof=json.loads(proof_path.read_text())
        if code_hash(executed)==code_hash(notebook)==proof['notebook_code_sha256']:notebook=executed
    existing.write_text(json.dumps(notebook,ensure_ascii=False,indent=1),encoding='utf-8')
    (directory/'answer.csv').write_bytes((ROOT/manifest['answer_file']).read_bytes())
    (directory/'README.md').write_text('''# Avito v10

Поместите три исходных Parquet рядом с Avito.ipynb. Установите requirements.txt
и выполните Restart Kernel → Run All. Все веса находятся в архиве; внешние API
не нужны. Описание и воспроизводимые рецептуры обучения: docs/EXPERIMENTS_V10.md.
Категория используется мягко; 0 не задаёт категорийное ограничение.
Большие обучающие кэши исключены. Финальный CSV проверяется по SHA-256.
''',encoding='utf-8')
    replacements={name:(directory/name).read_bytes() for name in ['Avito.ipynb','answer.csv','README.md']}
    for folder in ['.development','docs']:
        for path in (ROOT/folder).rglob('*'):
            if path.suffix in {'.py','.md'}:replacements[path.relative_to(ROOT).as_posix()]=path.read_bytes()
    replacements['docs/SOLUTION.md']=('> Базовое описание v7. В этом архиве v10; '
        'его изменения и измерения описаны в EXPERIMENTS_V10.md.\n\n'.encode()+replacements['docs/SOLUTION.md'])
    replacements['experiments/Avito_neural_experiments.ipynb']=(ROOT/'experiments/Avito_neural_experiments.ipynb').read_bytes()
    for key in ['history_bank','contextual_vectors','ranker','field_ranker','field_teacher_vectors','bge_queries','bge_ranker','warm_aux_ranker']:
        if manifest.get(key):replacements[manifest[key]]=(ROOT/manifest[key]).read_bytes()
    for details in manifest.get('field_documents',{}).values():
        replacements[details['path']]=(ROOT/details['path']).read_bytes()
    if manifest.get('bge_kind'):
        details=manifest['bge_documents']
        replacements[details['path']]=(ROOT/details['path']).read_bytes()
        for name in manifest['bge_checkpoint']['files_sha256']:
            path=ROOT/'models/bge-m3'/name
            replacements[path.relative_to(ROOT).as_posix()]=path.read_bytes()
        replacements['models/bge-m3/manifest.json']=(ROOT/'models/bge-m3/manifest.json').read_bytes()
    for path in manifest.get('cross_files_sha256',{}):replacements[path]=(ROOT/path).read_bytes()
    for path in (ROOT/'artifacts/quality-v10').glob('*.json'):
        # Training group metadata is reproducible, large, and unnecessary for inference.
        if '_training_' not in path.name and path.name!='bge_process.json':replacements[path.relative_to(ROOT).as_posix()]=path.read_bytes()
    for path in (ROOT/'artifacts/quality-v10').glob('*development.csv'):
        replacements[path.relative_to(ROOT).as_posix()]=path.read_bytes()
    errors=ROOT/'artifacts/quality-v10/development_misses.csv'
    if errors.exists():replacements[errors.relative_to(ROOT).as_posix()]=errors.read_bytes()
    for folder in ['warm-history-v10','semantic-fields-v10','field-ranker-v10','cross-finetune-v10','bge-m3-v10']:
        directory=ROOT/'artifacts'/folder
        if not directory.exists():continue
        for path in directory.iterdir():
            if path.suffix=='.json' and '_training_' not in path.name and '_mixed_' not in path.name or path.name in {'development.csv','joint_development.csv'}:
                replacements[path.relative_to(ROOT).as_posix()]=path.read_bytes()
    archive_path=ROOT/'deliverables/avito_v10_solution.zip'
    with zipfile.ZipFile(ROOT/'deliverables/avito_v9_solution.zip') as baseline:
        with zipfile.ZipFile(archive_path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as archive:
            for entry in baseline.infolist():
                if entry.filename not in replacements:archive.writestr(entry.filename,baseline.read(entry))
            for name,data in replacements.items():archive.writestr(name,data)
    print('Packaged v10',round(archive_path.stat().st_size/2**20,1),'MiB',flush=True)


if __name__=='__main__':package()
