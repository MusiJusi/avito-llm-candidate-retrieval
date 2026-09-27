"""Package a selected context candidate with every frozen inference dependency."""
from pathlib import Path
import ast
import hashlib
import json
import zipfile
ROOT=Path(__file__).resolve().parents[1]

def package():
    manifest=json.loads((ROOT/'artifacts/context-v9/manifest.json').read_text())
    notebook=json.loads((ROOT/'experiments/Avito_v9_candidate.ipynb').read_text(encoding='utf-8'))
    portable=dict(manifest);portable['answer_file']='answer.csv'
    notebook['cells'][0]['source']=['# Avito: кандидат v9\n',
        '\nRestart Kernel → Run All создаёт answer.csv без повторного обучения и внешних API.\n',
        'Подход, измерения и ограничения: docs/EXPERIMENTS_V9.md. Метрика платформы пока неизвестна.\n']
    for cell in notebook['cells']:
        if cell['cell_type']!='code':continue
        text=''.join(cell['source'])
        if text.startswith('FROZEN_QUERY_MANIFEST = '):
            text='FROZEN_QUERY_MANIFEST = '+repr(portable)+'\n'+text[text.index('\n')+1:]
        ast.parse(text);cell['source']=text.splitlines(True)
    directory=ROOT/'deliverables/v9';directory.mkdir(parents=True,exist_ok=True)
    existing=directory/'Avito.ipynb';proof=ROOT/'artifacts/context-v9/reproduction.json'
    def code_hash(document):return hashlib.sha256('\n'.join(''.join(c['source']) for c in document['cells'] if c['cell_type']=='code').encode()).hexdigest()
    if existing.exists() and proof.exists():
        executed=json.loads(existing.read_text(encoding='utf-8'))
        if code_hash(executed)==code_hash(notebook)==json.loads(proof.read_text())['notebook_code_sha256']:notebook=executed
    existing.write_text(json.dumps(notebook,ensure_ascii=False,indent=1),encoding='utf-8')
    (directory/'answer.csv').write_bytes((ROOT/manifest['answer_file']).read_bytes())
    description='''# Avito: кандидат v9

Архив содержит один основной `Avito.ipynb`, его `answer.csv` и локальные
зависимости инференса. Поместите три предоставленных Parquet рядом с notebook,
установите requirements.txt и выполните Restart Kernel → Run All. Обычный запуск
не обучает модели и не обращается к внешним inference API.

Описание v9 и его ограничений — docs/EXPERIMENTS_V9.md. Базовое устройство
лексического поиска, E5 и LambdaRank — docs/SOLUTION.md. Исторические сигналы
используют только предоставленный train. `search_category` вне scoring.

Векторы запросов и семантических соседей — сохранённый детерминированный результат
модельного расчёта; это не разметка benchmark. Исторический банк, модели, исходные
файлы и итоговый CSV проверяются по SHA-256. Обучение вынесено в .development и
experiments/Avito_neural_experiments.ipynb. Большие OOF-кэши в архив не входят;
для нового обучения сначала выполняются базовые рецептуры v7, затем v9.
Результат платформы этого кандидата пока неизвестен.
'''
    (directory/'README.md').write_text(description,encoding='utf-8')
    replacements={name:(directory/name).read_bytes() for name in ['Avito.ipynb','answer.csv','README.md']}
    for folder in ['.development','docs']:
        for path in (ROOT/folder).rglob('*'):
            if path.suffix in {'.py','.md'}:replacements[path.relative_to(ROOT).as_posix()]=path.read_bytes()
    replacements['CHANGELOG.md']=(ROOT/'CHANGELOG.md').read_bytes()
    replacements['docs/SOLUTION.md']=('> Базовое описание v7. Текущий архив содержит кандидат v9; '
        'его изменения и результаты — в EXPERIMENTS_V9.md.\n\n'.encode()+replacements['docs/SOLUTION.md'])
    research=ROOT/'experiments/Avito_neural_experiments.ipynb'
    replacements[research.relative_to(ROOT).as_posix()]=research.read_bytes()
    for key in ['history_bank','query_evidence','context_ranker']:
        if manifest.get(key):replacements[manifest[key]]=(ROOT/manifest[key]).read_bytes()
    for path in (ROOT/'artifacts/context-v9').glob('*'):
        if path.suffix in {'.json','.csv'}:replacements[path.relative_to(ROOT).as_posix()]=path.read_bytes()
    output=ROOT/'deliverables/avito_v9_solution.zip'
    with zipfile.ZipFile(ROOT/'deliverables/avito_v8_solution.zip') as old:
        with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as archive:
            for entry in old.infolist():
                if entry.filename not in replacements:archive.writestr(entry.filename,old.read(entry))
            for name,data in replacements.items():archive.writestr(name,data)
    print('Packaged context v9',round(output.stat().st_size/2**20,1),'MiB',flush=True)
if __name__=='__main__':package()
