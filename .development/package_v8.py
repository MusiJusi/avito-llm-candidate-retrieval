"""Create a portable v8 release with a single inference notebook and answer.

The confirmed v7 files remain in the workspace. A release reuses its immutable
offline dependencies, then replaces the entry point with the selected v8 model.
"""
from pathlib import Path
import ast
import hashlib
import json
import zipfile

ROOT=Path(__file__).resolve().parents[1]

def package():
    manifest=json.loads((ROOT/'artifacts/combined-pool-v8/manifest.json').read_text())
    notebook=json.loads((ROOT/'experiments/Avito_v8_candidate.ipynb').read_text(encoding='utf-8'))
    portable=dict(manifest);portable['answer_file']='answer.csv'
    notebook['cells'][0]['source']=['# Avito: кандидат v8\n',
        '\nRestart Kernel → Run All создаёт answer.csv.\n',
        'Локальное сравнение, описание подхода и ограничения: docs/EXPERIMENTS_V8.md.\n',
        'Метрика этого кандидата на платформе пока неизвестна.\n']
    for cell in notebook['cells']:
        if cell['cell_type']!='code':continue
        value=''.join(cell['source'])
        if value.startswith('FROZEN_QUERY_MANIFEST = '):
            value='FROZEN_QUERY_MANIFEST = '+repr(portable)+'\n'+value[value.index('\n')+1:]
        ast.parse(value);cell['source']=value.splitlines(True)
    directory=ROOT/'deliverables/v8'
    directory.mkdir(parents=True,exist_ok=True)
    executed_path=directory/'Avito.ipynb'
    proof=ROOT/'artifacts/combined-pool-v8/reproduction.json'
    if proof.exists() and executed_path.exists():
        executed=json.loads(executed_path.read_text(encoding='utf-8'))
        def code_hash(document):
            return hashlib.sha256('\n'.join(''.join(c['source']) for c in document['cells']
                if c['cell_type']=='code').encode()).hexdigest()
        reproduction=json.loads(proof.read_text())
        if code_hash(notebook)==code_hash(executed)==reproduction['notebook_code_sha256']:
            notebook=executed
    (directory/'Avito.ipynb').write_text(json.dumps(notebook,ensure_ascii=False,indent=1),encoding='utf-8')
    (directory/'answer.csv').write_bytes((ROOT/manifest['answer_file']).read_bytes())
    (directory/'README.md').write_text('''# Avito: кандидат v8

В архиве один основной notebook `Avito.ipynb` и его `answer.csv`.
Поместите три предоставленных Parquet рядом с notebook, установите
`requirements.txt`, выполните Restart Kernel → Run All. Сеть при воспроизведении
не требуется: веса, tokenizer и детерминированные векторы находятся в архиве.
Обычный запуск использует поставляемые веса; обучение не запускается автоматически.

Решение объединяет лексический поиск, исходную и дообученную multilingual-e5-small,
географию, историю и LightGBM. `search_category` в scoring не используется.
Описание признаков и исходного обучения — `docs/SOLUTION.md`; изменения v8,
локальные сравнения, неудачные пилоты и ограничения — `docs/EXPERIMENTS_V8.md`.
Обучение и исследования доступны в `experiments/Avito_neural_experiments.ipynb`
и `.development`. Использованы локальные open-source модели и библиотеки,
перечисленные в документации и requirements. Результат платформы v8 пока неизвестен.
''',encoding='utf-8')
    replacements={name:(directory/name).read_bytes() for name in ['Avito.ipynb','answer.csv','README.md']}
    replacements['CHANGELOG.md']=(ROOT/'CHANGELOG.md').read_bytes()
    # Only the research notebook belongs in this release; intermediate candidates
    # and their answers stay in the host workspace for comparison.
    for folder in ['.development','docs']:
        for path in (ROOT/folder).rglob('*'):
            if path.suffix in {'.py','.md'}:replacements[path.relative_to(ROOT).as_posix()]=path.read_bytes()
    replacements['docs/SOLUTION.md']=('> Этот документ описывает базовую v7. '
        'В текущем архиве основной notebook и answer.csv относятся к кандидату v8; '
        'его изменения и проверки описаны в EXPERIMENTS_V8.md.\n\n'.encode('utf-8')
        +replacements['docs/SOLUTION.md'])
    path=ROOT/'experiments/Avito_neural_experiments.ipynb'
    replacements[path.relative_to(ROOT).as_posix()]=path.read_bytes()
    for folder in ['ranker-v8','expanded-pool-v8','combined-pool-v8','learned-pool-v8',
                   'ce-feature-v8','query-hard-negative-v8']:
        for path in (ROOT/'artifacts'/folder).glob('*'):
            if path.suffix in {'.json','.csv'}:replacements[path.relative_to(ROOT).as_posix()]=path.read_bytes()
    replacements[manifest['final_ranker']]=(ROOT/manifest['final_ranker']).read_bytes()
    output=ROOT/'deliverables/avito_v8_solution.zip'
    with zipfile.ZipFile(ROOT/'deliverables/avito_solution.zip') as baseline:
        with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as archive:
            for entry in baseline.infolist():
                if entry.filename not in replacements and not (entry.filename.startswith('experiments/')
                        and entry.filename!='experiments/Avito_neural_experiments.ipynb'):
                    archive.writestr(entry.filename,baseline.read(entry))
            for name,data in replacements.items():archive.writestr(name,data)
    print('Packaged v8',output.name,round(output.stat().st_size/2**20,1),'MiB',flush=True)

if __name__=='__main__':package()
