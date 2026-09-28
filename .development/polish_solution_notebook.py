"""Turn the frozen, reproducible v11 notebook into a readable solution.

Only markdown, comments and the representation of constant manifests change.
The algorithmic code and the order in which it executes stay the same. The
manifest dictionaries move to a human-readable JSON file with a pinned SHA.
"""
from pathlib import Path
import ast
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / 'solution.ipynb'
CONFIG = ROOT / 'config/solution_v11.json'


def source(cell):
    return ''.join(cell['source'])


def markdown(text, identifier):
    return {'cell_type': 'markdown', 'id': identifier, 'metadata': {},
            'source': text.strip().splitlines(True)}


def assignment(cell, name):
    text = source(cell)
    node = ast.parse(text).body[0]
    assert isinstance(node, ast.Assign) and len(node.targets) == 1
    assert isinstance(node.targets[0], ast.Name) and node.targets[0].id == name
    return ast.literal_eval(node.value), ''.join(text.splitlines(True)[node.end_lineno:])


def rewrite():
    notebook = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
    assert len(notebook['cells']) == 47, 'Apply this transformation once to the frozen v11 notebook'
    cells = notebook['cells']
    frozen, remainder = assignment(cells[36], 'FROZEN_QUERY_MANIFEST')
    quality_v10, tail_v10 = assignment(cells[43], 'QUALITY_V10_MANIFEST')
    quality_v11, tail_v11 = assignment(cells[45], 'QUALITY_V11_MANIFEST')
    assert not tail_v10.strip() and not tail_v11.strip()
    assert quality_v11['answer_file'] == 'answer.csv'
    assert quality_v11['answer_sha256'] == hashlib.sha256((ROOT/'answer.csv').read_bytes()).hexdigest()
    CONFIG.parent.mkdir(parents=True, exist_ok=True)
    payload = {'retrieval': frozen, 'quality_v10': quality_v10,
               'quality_v11': quality_v11}
    CONFIG.write_text(json.dumps(payload, ensure_ascii=False, indent=2)+'\n',
                      encoding='utf-8')
    digest = hashlib.sha256(CONFIG.read_bytes()).hexdigest()
    cells[36]['source'] = ("SOLUTION_CONFIG_PATH = ROOT / 'config/solution_v11.json'\n"
                           f"assert sha256_file(SOLUTION_CONFIG_PATH) == '{digest}'\n"
                           "SOLUTION_CONFIG = json.loads(SOLUTION_CONFIG_PATH.read_text(encoding='utf-8'))\n"
                           "FROZEN_QUERY_MANIFEST = SOLUTION_CONFIG['retrieval']\n"
                           +remainder).splitlines(True)
    cells[43]['source'] = ["QUALITY_V10_MANIFEST = SOLUTION_CONFIG['quality_v10']\n"]
    cells[45]['source'] = ["QUALITY_V11_MANIFEST = SOLUTION_CONFIG['quality_v11']\n"]

    replacements = {
        0: '''# Решение: поиск 50 объявлений для запроса об услуге

**Итоговая версия v11.** Платформа: **Recall@50 = 0.914682**. Notebook
воспроизводит ровно тот `answer.csv`, который был отправлен. Порядок
объявлений внутри 50 на метрику не влияет.

Здесь показаны подготовка данных, защита валидации от утечки, каналы поиска,
признаки и применение сохранённых моделей. Код обучения моделей и протокол
экспериментов находятся в `.development/` и [docs/SOLUTION.md](docs/SOLUTION.md).
''',
        1: '''## Перед запуском

Положите `train.parquet`, `benchmark_queries.parquet` и
`benchmark_items.parquet` рядом с notebook. Установите `requirements.txt`.
Выполните **Restart Kernel → Run All**: notebook создаст `answer.csv` в той
же папке и проверит его SHA-256. Веса и готовые векторы включены в архив;
внешний API не требуется. CPU поддерживается, GPU ускоряет расчёт.

`train` содержит наблюдаемые пары «запрос — выбранное объявление».
`benchmark_queries` содержит запросы без меток, `benchmark_items` — корпус
для поиска. Идентификаторы всё время остаются строками. Метки benchmark
нигде не используются.
''',
        4: '''## 1. Валидация без утечки меток

Один и тот же короткий текст встречается в разных контекстах. Поэтому
разделение проводится по нормализованному тексту и контексту запроса, а
собственные взаимодействия контрольного запроса исключаются из истории.
Сначала строится неизменяемая таблица положительных пар, затем из доступной
истории создаются варианты с новой и знакомой формулировкой.
''',
        7: '''### Проверка полноты разметки и групп

Исходные положительные метки хранятся отдельно от очищенной истории: удаление
наблюдения из признаков не должно удалять его из контрольной разметки.
Этот инвариант исправил потерю семи положительных пар в ранней версии.
''',
        9: '''### Контексты обучения и контрольные режимы

Отдельно проверяются новые тексты запросов и известные тексты в новом
контексте. Локальная метрика считается на полном пуле кандидатов, а не на
подвыборке отрицательных примеров, которая использовалась для обучения.
''',
        11: '''## 2. Подготовка корпуса и быстрый поиск

Следующие ячейки задают рабочий индекс. Начальный аудит выше намеренно
загружает те же Parquet-файлы отдельно: он проверяет состав данных и
валидационные ограничения до построения поисковых структур. Повторная
инициализация ниже создаёт каноническое состояние для применения модели.
''',
        22: '''## 3. История, география и подкатегория

Для каждого кандидата считаются частота взаимодействий, уверенность истории,
связь запроса с подкатегорией и локацией, расстояние до исполнителя и
доступность выездной или дистанционной услуги. Пропуски остаются явными,
чтобы ранкер сам выбирал, насколько доверять каждому сигналу.
''',
        24: '''### Вектор признаков пары «запрос — объявление»

История добавляется к текстовым и географическим сигналам. Объявление,
которое не выбирали в логе, не считается доказанно нерелевантным.
''',
        26: '''## 4. Как обучались и проверялись ранкеры

Код ниже сохраняет рецепт обучения: все положительные пары и трудные
кандидаты формируют группы для LightGBM LambdaRank. Обычный запуск notebook
**не переобучает** модели: он использует локальные веса и векторы с
проверкой хешей. Это делает отправленный CSV воспроизводимым на CPU.
''',
        28: '''### Признаки, веса групп и LightGBM

Положительные пары имеют больший вес; отрицательные выбираются из
лексического, семантического и географического пулов. Отбор модели проведён
по Recall@50 на новых текстах и знакомых текстах с новым контекстом.
''',
        30: '''### Слияние оценок и локальная метрика

Ранги из нескольких каналов объединяются reciprocal rank fusion (RRF).
Предел в 50 объявлений применяется после объединения всех сигналов.
''',
        32: '''### Рецепт обучения и экспорт весов

Функции оставлены в notebook, чтобы проверяющий видел путь обучения.
Выполняемый в конце экспорт обращается к сохранённым финальным весам.
''',
        35: '''## 5. Зафиксированные компоненты итогового каскада

Большие конфигурационные словари вынесены в `config/solution_v11.json` для
удобства чтения. Хеш конфигурации проверяется до применения. Все модели,
векторы и корпусные справочники читаются локально; существенные файлы также
проверяются по SHA-256 перед использованием.
''',
        41: '''### Контекстные признаки и обученный query encoder

Обученный query encoder E5 сопоставляет запрос с объявлением, а история и
соседние запросы добавляют сигналы подкатегории и локации. Документный
encoder остаётся замороженным, поэтому векторы корпуса можно переиспользовать.
''',
        42: '''### Базовый ранкер v10

Базовая модель учитывает лексические и E5-признаки, историю, категорию,
географию, поля объявления и BGE-M3. Она служит также вторым членом
финальной смеси: новый ранкер не заменяет её полностью.
''',
        44: '''### Добавленные признаки v11

Десять новых признаков проверяют явный вид и тип услуги, распространённость
объявлений в локации, возможность выезда или удалённой работы и их связь с
расстоянием. Они улучшают отбор без жёсткого удаления кандидатов. Новый
LightGBM обучен на 39,67 млн пар; его ранг смешивается с рангом v10 50/50.
''',
    }
    for index, text in replacements.items():
        cells[index]['source'] = text.strip().splitlines(True)

    additions = {
        16: '''### Лексический индекс

Русская нормализация, стемминг и BM25 дают быстрый канал для точных
совпадений. Фильтры запроса индексируются отдельно от основного текста.
''',
        17: '''### Корпусные массивы и географическая совместимость

Числовые поля и координаты преобразуются в плотные массивы, чтобы тысячи
объявлений одного запроса оценивались пакетно.
''',
        19: '''### Признаки и набор трудных конкурентов

Текстовое покрытие, цена, рейтинг, расстояние, ранги поиска и источник
попадания в пул передаются ранкеру. Случайные кандидаты сохраняют широту
обучения, трудные приближают его к границе top-50.
''',
        20: '''### Семантический поиск E5

Локальный encoder сопоставляет короткий запрос с названием, параметрами и
описанием объявления. Векторы документов сохранены; код не обращается к
сетевым сервисам при воспроизведении.
''',
        21: '''### Объединение каналов кандидатного пула

Лексические и семантические находки дополняют друг друга. Полнота широкого
пула на локальной валидации около 0.995; объявление вне пула ранкер вернуть
не сможет.
''',
        36: '''### Манифест поиска и проверка файлов

В конфигурации записаны пути, версии, параметры модели и хеши артефактов.
Загрузка не зависит от конкретного `query_id` и не содержит тестовой разметки.
''',
        38: '''### Кодирование запросов

Функция ниже нужна, если готовые векторы запросов не найдены. Она работает
локально и использует те же замороженные веса.
''',
        40: '''### Подбор соседних запросов и контекстное расширение

Соседи дают дополнительную вероятность подкатегории, но итоговый выбор
остаётся за ранкером, который видит также прямые признаки объявления.
''',
        46: '''## 6. Генерация и проверка `answer.csv`

Эта функция загружает сохранённые векторы и модели, строит полный пул,
считает признаки v10 и v11, объединяет ранги и сохраняет до 50 строковых
`item_id` на запрос. В конце проверяются количество строк, уникальность,
формат и принадлежность корпусу, затем SHA-256 всего CSV.
''',
    }
    final = source(cells[46])
    comments = {
        "    vector_path=ROOT/FROZEN_QUERY_MANIFEST['query_vectors']\n":
            '    # 1. Запросные векторы: готовый локальный кэш или локальный encoder.\n',
        "    # The frozen bank contains training aggregates, not benchmark answers.\n":
            '    # 2. Исторические агрегаты обучения и географические соседи.\n',
        "    pool_path=V5_CACHE/f'expanded_benchmark_{FROZEN_QUERY_MANIFEST[\"fingerprint\"]}.joblib'\n":
            '    # 3. Широкий лексико-семантический пул и базовые признаки.\n',
        "    original=baseline_v5(records,features,final=True)\n":
            '    # 4. Базовый скоринг с обученным query encoder.\n',
        '    context_matrices=[]\n':
            '    # 5. Контекстные, географические и соседские признаки.\n',
        '    from quality_model_v10 import adjust_quality_scores\n':
            '    # 6. Финальный ранкер v10 на текстовых, исторических и BGE-признаках.\n',
        '    from quality_model_v11 import adjust_v11_scores\n':
            '    # 7. Структурный ранкер v11 и слияние рангов 50/50.\n',
        '    predictions=top50(records,score)\n':
            '    # 8. Выбор top-50 и проверка идентификаторов до сохранения.\n',
        "    output=ROOT/FROZEN_QUERY_MANIFEST['answer_file']\n":
            '    # 9. Побайтовая проверка против отправленного файла.\n',
    }
    for anchor, comment in comments.items():
        assert final.count(anchor) == 1, anchor
        final = final.replace(anchor, comment+anchor)
    final = final.replace("print('Validated neural candidate:'",
                          "print('Validated solution:'")
    ast.parse(final)
    cells[46]['source'] = final.splitlines(True)

    result = []
    for index, cell in enumerate(cells):
        if index in additions:
            result.append(markdown(additions[index], f'solution-v11-explain-{index}'))
        if cell['cell_type'] == 'code':
            ast.parse(source(cell))
            cell['execution_count'] = None
            cell['outputs'] = []
        result.append(cell)
    notebook['cells'] = result
    NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding='utf-8')
    print('Polished solution notebook:', len(result), 'cells; config SHA-256:', digest)


if __name__ == '__main__':
    rewrite()
