"""Create the reproducible v11 notebook from the frozen v10 notebook.

The last code cell reproduces the complete v10 pool, then applies the chosen
v11 ranker. A pending answer hash permits one local export before freezing.
"""
from pathlib import Path
import ast
import json

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'artifacts/bge-m3-v11/manifest.json'


def cell(kind, source, identifier):
    result = {'cell_type': kind, 'id': identifier, 'metadata': {},
              'source': source.splitlines(True)}
    if kind == 'code':
        result.update(execution_count=None, outputs=[])
    return result


def build():
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    path = ROOT / 'experiments/Avito_v10_candidate.ipynb'
    notebook = json.loads(path.read_text(encoding='utf-8'))
    notebook['cells'][0]['source'] = [
        '# Avito v11: признаки фильтров, географии и отбор трудных кандидатов\n',
        '\n',
        'Notebook воспроизводит локально выбранный вариант и сохраняет answer.csv.\n',
        'Описание данных, обучения и проверки: docs/EXPERIMENTS_V11.md.\n',
    ]
    for entry in notebook['cells']:
        if entry['cell_type'] == 'code':
            entry['execution_count'] = None
            entry['outputs'] = []
    explanation = '''## Дополнительный отбор v11

В новом ранкере отдельно измеряются соответствие вида и типа услуги фильтрам,
плотность объявлений в локации и признаки выездной или дистанционной работы.
В ходе разработки также проверялись относительные оценки BGE-M3 по всему
корпусу и обучение на трудных кандидатах. Сохраняются только варианты, которые
показали пользу на локальной валидации.

Обучение и повторная оценка находятся в `.development/quality_v11.py`;
этот notebook применяет сохранённые локальные веса к полному пулу и проверяет
формат CSV. Никаких внешних API при запуске не требуется.
'''
    notebook['cells'].insert(-1, cell('markdown', explanation, 'avito-v11-method'))
    notebook['cells'].insert(-1, cell('code',
        'QUALITY_V11_MANIFEST = '+repr(manifest)+'\n', 'avito-v11-manifest'))
    last = notebook['cells'][-1]
    source = ''.join(last['source'])
    anchor = '    predictions=top50(records,score)\n'
    assert source.count(anchor) == 1
    hook = '''    from quality_model_v11 import adjust_v11_scores
    score=adjust_v11_scores(ROOT,QUALITY_V10_MANIFEST,QUALITY_V11_MANIFEST,
        queries,items,records,context_matrices,score,semantic_index.items,
        sha256_file,blend_scores,predict_scores)
'''
    source = source.replace(anchor, hook+anchor)
    call = 'query_answer=export_frozen_query_candidate()\n'
    assert source.count(call) == 1
    source = source.replace(call,
        "FROZEN_QUERY_MANIFEST['answer_file']=QUALITY_V11_MANIFEST['answer_file']\n"
        "FROZEN_QUERY_MANIFEST['answer_sha256']=QUALITY_V11_MANIFEST['answer_sha256']\n"
        +call)
    old_assert = "    assert sha256_file(output)==FROZEN_QUERY_MANIFEST['answer_sha256']\n"
    assert source.count(old_assert) == 1
    source = source.replace(old_assert,
        "    if FROZEN_QUERY_MANIFEST['answer_sha256'] is not None:\n"
        "        assert sha256_file(output)==FROZEN_QUERY_MANIFEST['answer_sha256']\n")
    output_line = "    output=ROOT/FROZEN_QUERY_MANIFEST['answer_file']\n"
    assert source.count(output_line) == 1
    source = source.replace(output_line, output_line+
                            "    output.parent.mkdir(parents=True,exist_ok=True)\n")
    ast.parse(source)
    last['source'] = source.splitlines(True)
    output = ROOT / 'experiments/Avito_v11_candidate.ipynb'
    output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1),
                      encoding='utf-8')
    print('Notebook ready', output)


if __name__ == '__main__':
    build()
