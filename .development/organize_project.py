"""Promote the platform-confirmed v7; archive prior files without deleting data.

The numerical training sources are deliberately preserved byte for byte because
their SHA-256 values identify fitted models and expensive OOF training caches.
All moves are confined to the project and refuse to overwrite an archive entry.
"""
from pathlib import Path
import ast
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = 'a256639b72ef3bd515bf5118c5e337aee23a34c9e0b261da580b1c852411a6b2'

def move(name, directory):
    source = ROOT / name
    target = ROOT / directory / source.name
    assert source.resolve().is_relative_to(ROOT) and target.resolve().is_relative_to(ROOT)
    if source.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if source.is_file() and target.is_file() and source.read_bytes() == target.read_bytes():
                source.unlink()  # Resume a copy whose original removal was interrupted.
                return
            raise FileExistsError(target)
        shutil.move(str(source), str(target))

if __name__ == '__main__':
    candidate = ROOT / 'answer_query_encoder.csv'
    if not candidate.exists():
        candidate = ROOT/'archive/answers/answer_query_encoder.csv'
    assert hashlib.sha256(candidate.read_bytes()).hexdigest() == EXPECTED
    notebook_path = ROOT/'Avito_query_encoder_candidate.ipynb'
    if not notebook_path.exists():
        notebook_path = ROOT/'archive/notebooks/Avito_query_encoder_candidate.ipynb'
    notebook = json.loads(notebook_path.read_text(encoding='utf-8'))
    details_path = ROOT/'README_query_encoder.md'
    if not details_path.exists():
        details_path = ROOT/'archive/documentation/README_query_encoder.md'
    details = details_path.read_text(encoding='utf-8')
    original_sources = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
        [ROOT/'.development'/name for name in ['ranking_v5.py', 'validation_protocol.py',
        'microcat_v6.py', 'microcat_models.py', 'query_encoder_pilot.py',
        'query_encoder_rank.py', 'export_query_encoder_candidate.py']]}
    # Preserve the exact submitted notebook before changing its output filename.
    for path in sorted(ROOT.glob('Avito*.ipynb')):
        destination = 'experiments' if path.name == 'Avito_neural_experiments.ipynb' else 'archive/notebooks'
        move(path.name, destination)
    for path in sorted(ROOT.glob('answer*.csv')):
        move(path.name, 'archive/answers')
    for path in sorted(ROOT.glob('README*.md')):
        move(path.name, 'archive/documentation')
    for name in ['IMPROVEMENT_PLAN.md', 'IMPROVEMENT_PROGRESS.md']:
        move(name, 'docs/history')
    for name in ['initial_eda.json', 'data_profile.json', 'inspect_data.py']:
        move(name, 'archive/eda')
    for path in sorted((ROOT/'deliverables').glob('*.zip')):
        move(path.relative_to(ROOT).as_posix(), 'archive/releases')
    move('deliverables/query_encoder_submission/answer.csv', 'archive/submitted')
    old_submission = ROOT/'deliverables/query_encoder_submission'
    if old_submission.is_dir() and not any(old_submission.iterdir()):
        old_submission.rmdir()  # Remove only the now-empty directory.

    notebook['cells'][0]['source'] = [
        '# Основное решение Avito: дообученный query-encoder E5 и LambdaRank\n',
        '\nПлатформа: **Recall@50 = 0.897443**.\n',
        '\nRestart Kernel → Run All создаёт **answer.csv** в папке проекта.\n',
        'Обычный запуск использует сохранённые веса и работает локально без API.\n',
        'Описание, обучение и ограничения проверки: README.md и docs/SOLUTION.md.\n']
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if source.startswith('FROZEN_QUERY_MANIFEST = '):
                source = source.replace("'answer_file': 'answer_query_encoder.csv'", "'answer_file': 'answer.csv'")
                assert "'answer_file': 'answer.csv'" in source
            cell['source'] = source.splitlines(True)
            cell['execution_count'] = None
            cell['outputs'] = []
            ast.parse(source)
    notebook['metadata']['platform_recall50'] = 0.897443
    (ROOT/'Avito.ipynb').write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding='utf-8')
    shutil.copyfile(ROOT/'archive/answers/answer_query_encoder.csv', ROOT/'answer.csv')

    research_path = ROOT/'experiments/Avito_neural_experiments.ipynb'
    research = json.loads(research_path.read_text(encoding='utf-8'))
    research['cells'][0]['source'] = ['# Эксперименты и обучение\n',
        '\nОсновной notebook — ../Avito.ipynb. Этот файл показывает отчёты;\n',
        'дорогие эксперименты включаются явно через RUN_* = True.\n',
        'Исходные данные и модели находятся в родительской папке проекта.\n',
        'Исследовательские ответы сравниваем до замены основного answer.csv.\n']
    for cell in research['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            source = source.replace('ROOT = Path.cwd()',
                "ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents] if (p/'train.parquet').exists()), Path.cwd())")
            cell['source'] = source.splitlines(True)
            cell['execution_count'] = None
            cell['outputs'] = []
            ast.parse(source)
    research_path.write_text(json.dumps(research, ensure_ascii=False, indent=1), encoding='utf-8')

    (ROOT/'docs').mkdir(exist_ok=True)
    # Retain full technical explanation, replacing the obsolete release introduction.
    details = '# Описание основного решения\n\nПлатформа: **Recall@50 = 0.897443**. Основной notebook — `Avito.ipynb`; ответ — `answer.csv`.\n\n' + details[details.index('| Локальная проверка'):]
    details = details.replace('deliverables/avito_query_encoder_solution.zip', 'deliverables/avito_solution.zip')
    details = details.replace('Avito_query_encoder_candidate.ipynb', 'Avito.ipynb')
    details = details.replace('Avito_neural_experiments.ipynb', 'experiments/Avito_neural_experiments.ipynb')
    details = details.replace('`IMPROVEMENT_PROGRESS.md`', '`docs/history/IMPROVEMENT_PROGRESS.md`')
    details = details.replace('263.26 секунд.', '263.26 секунд до переноса файлов; повторная проверка нового основного notebook — artifacts/current_reproduction.json.')
    (ROOT/'docs/SOLUTION.md').write_text(details, encoding='utf-8')
    current = {'version': 'v7', 'platform_recall50': .897443, 'platform_result_source': 'user_reported',
        'notebook': 'Avito.ipynb', 'answer_file': 'answer.csv', 'answer_sha256': EXPECTED,
        'training_manifest': 'artifacts/query-encoder-candidate/manifest.json',
        'original_submitted_notebook': 'archive/notebooks/Avito_query_encoder_candidate.ipynb',
        'immutable_training_source_sha256': original_sources}
    (ROOT/'artifacts/current_solution.json').write_text(json.dumps(current, indent=2), encoding='utf-8')
    for name, digest in original_sources.items():
        assert hashlib.sha256((ROOT/'.development'/name).read_bytes()).hexdigest() == digest
    print('Promoted platform-confirmed v7; archived old notebooks, answers, documentation and packages.')
