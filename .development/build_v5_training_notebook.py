"""Embed complete validation, index initialization and v5 training into a notebook."""
import ast
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
old=json.loads((ROOT/'Avito_v0.3.ipynb').read_text(encoding='utf-8'))
codes=[''.join(c['source']) for c in old['cells'] if c['cell_type']=='code']
validation=json.loads((ROOT/'Avito_validation_v5.ipynb').read_text(encoding='utf-8'))
pipeline=(ROOT/'.development/ranking_v5.py').read_text(encoding='utf-8')
pipeline_hash=hashlib.sha256(pipeline.encode('utf-8')).hexdigest()
protocol_hash=hashlib.sha256((ROOT/'.development/validation_protocol.py').read_bytes()).hexdigest()
old_code_hash=hashlib.sha256('\n'.join(codes).encode('utf-8')).hexdigest()
cells=[]


def add(kind,source):
    source=source.strip()+'\n'
    entry={'cell_type':kind,'metadata':{},'source':source.splitlines(True),
           'id':hashlib.sha256((kind+source).encode()).hexdigest()[:12]}
    if kind=='code':
        entry.update(execution_count=None,outputs=[])
    cells.append(entry)


def definitions(source,names=None):
    return '\n\n'.join(ast.get_source_segment(source,n) for n in ast.parse(source).body
        if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and (names is None or n.name in names))


add('markdown','''# Обучение отбора кандидатов v5

Самостоятельный notebook: Restart Kernel → Run All. Нужны три исходных Parquet,
зависимости requirements.txt и приложенные локальные модели. Внешних API нет.
Код подготовки gold, истории, индексов, обучения и вывода находится здесь.
`search_category` не используется в scoring.

Сравниваем широкие группы новых текстов, смешанные режимы истории и дополнительное
обучение на трудных отрицательных примерах текущего нового ранкера. Контрольные
600 текстов открываются после выбора. Замороженная v4 — внешний comparator;
её вспомогательная история могла содержать контрольные взаимодействия, поэтому
сравнение с ней не является полностью независимой оценкой всего старого пайплайна.
Новый ранкер обучается без контрольных взаимодействий и без оценок v4 как признаков.

Артефакты подготовки: artifacts/validation-v5. Обучение: artifacts/ranking-v5.
Ответ заменяется после выбора по development, финального обучения и проверки CSV.
Предыдущая отправка сохраняется отдельно в answer_v0.4.csv.''')
for c in validation['cells']:
    add(c['cell_type'],''.join(c['source']))
add('markdown','''## Локальные индексы и исходные 38 признаков

BM25 заголовка/текста и символьный TF-IDF вместе с замороженной multilingual E5
создают пул. Код исходных индексов сохранён из предыдущей версии; их кеши
переиспользуются только при совпадении входов и фиксированной версии рецептуры.
Дополнительные статистики обучения ниже имеют отдельный fingerprint.''')
for index in range(6):
    source=codes[index]
    if index==4:
        start=source.index('notebook_document = ')
        end=source.index('fingerprint = ',start)
        source=source[:start]+'# Frozen recipe identity for unchanged unsupervised indexes.\ncode_hash = '+repr(old_code_hash)+'\n'+source[end:]
    add('code',source)
add('code',definitions(codes[6])+"\nbest_config = {'title':.3,'body':.5,'char':.2,'filter':.03,'geo':.5,'micro':.5}\n")
add('code',codes[8].split('training_data = build_oof_training')[0]+'\n'+definitions(codes[9]))
add('code',codes[12])
add('code','''legacy_retrieve_features = retrieve_features
legacy_rank_features = rank_features
legacy_score_candidates = score_candidates
legacy_negative_sample = hard_negative_sample
LEGACY_FEATURE_NAMES = list(RANK_FEATURE_NAMES)
SEMANTIC_FEATURE_NAMES = ['e5_cosine','e5_geo_affinity','from_lexical_pool','from_semantic_pool','log_e5_rank','log_e5_geo_rank']
RANK_FEATURE_NAMES += SEMANTIC_FEATURE_NAMES
'''+definitions(codes[13])+'''
retrieve_features = retrieve_semantic_features
rank_features = rank_semantic_features
score_candidates = score_lexical_columns
hard_negative_sample = semantic_negative_sample
''')
add('markdown','''## Конфигурация нового обучения

Полные метки и разбиения созданы выше из исходного train. Никакие development- или
контрольные тексты не входят в новое прямое обучение. Трудные negatives ищет новый
ранкер, обученный на этих же разрешённых данных; старые priors для этого не нужны.
При расширении финального обучения допускаем все размеченные контексты корпуса.''')
source=pipeline[pipeline.index('INDEX_CACHE = CACHE'):pipeline.index('class V5History')]
source=source.replace("hashlib.sha256((ROOT/'.development/validation_protocol.py').read_bytes()).hexdigest()",repr(protocol_hash))
source=source.replace('hashlib.sha256(Path(__file__).read_bytes()).hexdigest()',repr(pipeline_hash))
add('code',"from types import MappingProxyType\n# Use the project's local dependency folder when present; otherwise requirements.txt.\nif (ROOT / '.ranking_deps').is_dir():\n    sys.path.insert(0, str(ROOT / '.ranking_deps'))\nimport lightgbm as lgb\n"+source)
groups=[(['V5History','v5_features'],'''## Уверенность истории

К исходным признакам добавлены факт и объём истории текста, энтропия и пик
распределения услуг, наличие координат, вероятность и поддержка перехода локаций.
Каждый признак использует только очищенную историю данного фолда.'''),
(['sample_v5_group','prepare_v5_training'],'''## Полные пулы и обучающие группы

Сохраняем полученные без labels пулы, затем выбираем извлечённые positives,
лексические/семантические лидеры и случайный хвост. Во втором проходе добавляем
высоко оценённые новым ранкером negatives. Пропущенные positives не внедряем.
Веса групп согласуют фактическую долю известных текстов с benchmark.'''),
(['fit_v5','evaluation_features','predict_scores','top50','old_v4_scores'],'''## Обучение и сравнение

LightGBM LambdaRank, 31 лист, максимум 400 деревьев. Группы остаются непрерывными.
Старый ансамбль нужен для сравнения и проверки сочетания ошибок. Его scores не
являются входами новой модели. Оценочные истории исключают метки другой выборки.'''),
(['blend_scores','matched_slice_recall','development_metrics'],'''## Выбор по Recall@50

Проверяем префиксы 100/200/400 деревьев и веса RRF 50/75/100% нового ранкера.
В известном режиме учитываем только фактически известные тексты. Внутри срезов
согласуем длину текста, фильтры и наличие центра локации с benchmark без labels.
Улучшение агрегата не должно сопровождаться падением новых текстов более 0.5 п.п.'''),
(['main','final_fit_and_export'],'''## Запуск, контроль и финальный ответ

Выбор фиксируется до оценки 600 контрольных текстов. После него обучаем финальную
модель на всех доступных положительных контекстах и проверяем идентификаторы,
колонки, число строк, уникальность кандидатов и повторяемость предсказаний.
Если на development выигрывает исходная v4, оставляем её ответ.''')]
for names,description in groups:
    add('markdown',description)
    source=definitions(pipeline,set(names)).replace('hashlib.sha256(Path(__file__).read_bytes()).hexdigest()',repr(pipeline_hash))
    add('code',source)
add('code','main()')
notebook={'nbformat':4,'nbformat_minor':5,'cells':cells,'metadata':{
    'kernelspec':{'display_name':'Python 3 (ipykernel)','language':'python','name':'python3'},
    'language_info':{'name':'python','version':'3.14.6'},'pipeline_sha256':pipeline_hash}}
(ROOT/'Avito_training_v5.ipynb').write_text(json.dumps(notebook,ensure_ascii=False,indent=1),encoding='utf-8')
print('Created self-contained v5 training notebook:',len(cells),'cells; pipeline',pipeline_hash)
