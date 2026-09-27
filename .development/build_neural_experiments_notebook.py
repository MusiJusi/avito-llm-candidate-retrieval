"""A Jupyter entry point for documented local pilots and their actual reports."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]
cells=[]
def md(value):cells.append({'cell_type':'markdown','metadata':{},'source':value.splitlines(True)})
def code(value):cells.append({'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':value.splitlines(True)})
md('''# Локальные нейросетевые и географические эксперименты

Основное воспроизводимое решение — `Avito.ipynb`. Этот notebook читает реальные
отчёты и позволяет повторить отдельные эксперименты из сопровождаемого кода
в `.development`. Каждый запуск работает локально, без inference API.

По умолчанию дорогое обучение выключено. Для повторения выбери соответствующие
значения `RUN_* = True`. Файлы Parquet должны лежать рядом с notebook.
`answer.csv` эти исследования не изменяют. Полный журнал ограничений —
`IMPROVEMENT_PROGRESS.md`; экспериментальный microcat-ответ воспроизводится
отдельным `Avito_microcat_candidate_v6.ipynb`.
''')
code('''from pathlib import Path
import json
import runpy
ROOT = Path.cwd()
assert (ROOT / 'train.parquet').exists(), 'Открой notebook в папке проекта'
RUN_CROSS_ENCODER = False
RUN_QUERY_ENCODER = False
RUN_QUERY_RANKER = False
RUN_SERVICE_GEOGRAPHY = False
RUN_COLD_SENSITIVITY = False
RUN_FINAL_QUERY_REFIT = False
RUN_BLEND_CHECK = False
RUN_QUERY_ERROR_ANALYSIS = False

def show_report(folder, filename):
    path = ROOT / 'artifacts' / folder / filename
    if path.exists():
        print(path.relative_to(ROOT))
        print(json.dumps(json.loads(path.read_text(encoding='utf-8')), ensure_ascii=False, indent=2))
    else:
        print('Эксперимент ещё не завершён:', path.relative_to(ROOT))

def run_local(name):
    # run_name='__main__' сохраняет совместимость локальных pickle/joblib-классов.
    runpy.run_path(str(ROOT / '.development' / name), run_name='__main__')
''')
md('''## Frozen cross-encoder

`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` совместно кодирует запрос и
объявление из первых 300/500 результатов v5. Проверяем только development:
готовый текстовый score может не соответствовать фактическому выбору услуги.
Модель скачивается один раз; ревизия и SHA256 проверяются downloader-ом.
Проверка прямой смеси не заменяет обучение признака внутри ранкера.
''')
code('''if RUN_CROSS_ENCODER:
    if not (ROOT/'models/mmarco-miniLM-cross-encoder/manifest.json').exists():
        run_local('download_cross_encoder.py')
    run_local('cross_encoder_pilot.py')
show_report('cross-encoder-pilot', 'pilot_report.json')
''')
md('''## Query-encoder E5 с замороженными документами

Используем разрешённые пары, включая объявления вне benchmark-корпуса.
Контрастивные minibatch учитывают несколько известных положительных документов;
регуляризация к исходным query-векторам ограничивает дрейф. Пилот исключает
все тексты development и контроля из своей обучающей истории.
Подбор смеси идёт по development; затем параметры фиксируются для контроля.
''')
code('''if RUN_QUERY_ENCODER:
    run_local('query_encoder_pilot.py')
    run_local('query_encoder_control.py')
show_report('query-encoder-pilot', 'training_report.json')
show_report('query-encoder-pilot', 'pilot_report.json')
show_report('query-encoder-pilot', 'control_report.json')
''')
md('''## Обучаемый отбор с новым E5-сигналом

Шесть независимых query-encoder соответствуют трём фолдам в двух режимах
истории. Собственные контексты исключаются; в режиме новых текстов исключается
весь текст запроса. LambdaRank получает OOF-оценки, поэтому обучающая пара
не используется для обучения энкодера, который считает её признак.
Признаки: cosine, cosine с географией, изменения относительно исходной E5
и логарифмы рангов внутри полного пула. Группы и отрицательные примеры равны v5.
''')
code('''if RUN_QUERY_RANKER:
    run_local('query_encoder_rank.py')
show_report('query-encoder-rank', 'selection.json')
show_report('query-encoder-rank', 'control.json')
if RUN_BLEND_CHECK:
    run_local('query_micro_blend.py')
show_report('query-micro-blend', 'report.json')
if RUN_QUERY_ERROR_ANALYSIS:
    run_local('query_rank_error_analysis.py')
show_report('query-encoder-rank', 'error_summary.json')
if RUN_FINAL_QUERY_REFIT:
    run_local('export_query_encoder_candidate.py')
show_report('query-encoder-candidate', 'manifest.json')
show_report('query-encoder-candidate', 'portable_reproduction.json')
''')
md('''## География по виду услуги

Оцениваем вероятность совпадения локации и переходов между локациями отдельно
для microcat. Малые выборки сглаживаются глобальной статистикой. Эти признаки
передаются ранкеру; жёсткого географического фильтра нет. OOF-история совпадает
с историей v5 и исключает собственные метки.
''')
code('''if RUN_SERVICE_GEOGRAPHY:
    run_local('service_geo_experiment.py')
show_report('service-geography-v6', 'selection.json')
show_report('service-geography-v6', 'control.json')
''')
md('''## Чувствительность к доле новых положительных объявлений

На фиксированных 400 development-контекстах изменяется доля собственных
положительных items, удаляемых из истории: 0%, 50%, 90%, 100%. Метки остаются
полными; веса ранкера заморожены. Это ограниченная диагностическая выборка,
она не доказывает независимость результата бенчмарка от cold items.
''')
code('''if RUN_COLD_SENSITIVITY:
    run_local('check_cold_sensitivity.py')
show_report('cold-sensitivity-v5', 'report.json')
''')
notebook={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
    'language_info':{'name':'python','version':'3.14.6'}},'nbformat':4,'nbformat_minor':5}
(ROOT/'Avito_neural_experiments.ipynb').write_text(json.dumps(notebook,ensure_ascii=False,indent=1),encoding='utf-8')
print('Created local experiment notebook.')
