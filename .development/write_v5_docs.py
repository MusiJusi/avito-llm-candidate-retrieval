"""Record results, frozen-component provenance and reviewer reproduction steps."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
CACHE=ROOT/'artifacts/ranking-v5'
manifest=json.loads((CACHE/'manifest.json').read_text(encoding='utf-8'))
selection=manifest['selection']; winner=selection['winner']; baseline=selection['baseline']
control=manifest['control']; fp=manifest['fingerprint']
initial=json.loads((CACHE/'evaluation_initial_report.json').read_text())
mined=json.loads((CACHE/'evaluation_mined_report.json').read_text())
final=json.loads((CACHE/'final_mined_report.json').read_text())
assert hashlib.sha256((ROOT/'answer.csv').read_bytes()).hexdigest()==manifest['answer_sha256']
components=[ROOT/'models/retrieval-priors'/name for name in
    ['evaluation_legacy.joblib','evaluation_semantic.joblib','final_legacy.joblib','final_semantic.joblib']]
components+=[ROOT/'artifacts/ranking-v1'/name for name in
    ['lgb_rank_expanded_200_c0442eeba2d244f2.joblib','final_ranker_c0442eeba2d244f2.joblib']]
components+=[CACHE/f'mixed_wide_{fp}.joblib']
weights={'input_sha256':manifest['input_sha256'],
    'frozen_components':{path.relative_to(ROOT).as_posix():hashlib.sha256(path.read_bytes()).hexdigest() for path in components},
    'final_model':{manifest['final_model']:hashlib.sha256((CACHE/manifest['final_model']).read_bytes()).hexdigest()},
    'embedding_sha256':{path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in CACHE.glob('e5_*.npy')},
    'own_mining_pilot':{'file':f'mixed_wide_{fp}.joblib','trained_on_provided_train':True,
        'contexts':19031,'trees':400,'predicted_prefix_for_mining':200,
        'development_and_control_labels_excluded':True,'role':'negative selection only, no inference feature'}}
(CACHE/'weights_manifest.json').write_text(json.dumps(weights,indent=2),encoding='utf-8')
ci=control['paired_delta_bootstrap95']
portable_path=CACHE/'portable_reproduction.json'
portable=json.loads(portable_path.read_text()) if portable_path.exists() else None
refit_path=CACHE/'refit_reproduction.json'
refit=json.loads(refit_path.read_text()) if refit_path.exists() else None
refit_description=(f"Повторный fit выбранной финальной модели с теми же подготовленными данными, без повторного использования "
    f"обученных весов, дал идентичные деревья и битовые оценки на {refit['benchmark_pairs_compared']:,} "
    f"парах benchmark за {refit['elapsed_seconds']:.2f} с. Это проверка детерминированности обучения; "
    "подготовленная выборка использована повторно." if refit else '')
reproduction=('Извлечённый архив проверен на CPU с заблокированными сетевыми соединениями: '
              f'те же байты answer.csv за {portable["elapsed_seconds"]:.2f} с.' if portable else
              'Изолированная проверка архива на CPU выполняется после упаковки.')
text=f'''# Кандидатогенерация услуг Авито, v0.5.0

Самостоятельный Jupyter Notebook **Avito.ipynb** создаёт кандидатов с помощью BM25,
символьного TF-IDF и локальной multilingual E5, затем отбирает 50 объявлений.
Новая модель LightGBM LambdaRank использует два режима истории и дополнительные
трудные отрицательные примеры. Итог — RRF: 75% нового ранкера, 25% ансамбля v4.
Все вычисления локальны, внешних API нет. `search_category` не входит в scoring.

## Результаты до новой отправки

| Проверка | v4, по тому же протоколу | v5 |
|---|---:|---:|
| Development: согласование распределений | {baseline['matched_recall50']:.6f} | **{winner['matched_recall50']:.6f}** |
| Development: новые тексты, macro Recall@50 | {baseline['unseen_macro_recall50']:.6f} | **{winner['unseen_macro_recall50']:.6f}** |
| Development: фактически известные тексты | {baseline['actual_known_macro_recall50']:.6f} | **{winner['actual_known_macro_recall50']:.6f}** |
| Контрольные 600 текстов | {control['v4_recall50']:.6f} | **{control['selected_recall50']:.6f}** |
| Закрытая платформа | **0.882692** | Пока не отправлена |

Development — 3400 ранее просмотренных контекстов, два режима тех же запросов.
Это не 6800 независимых наблюдений. Контрольные 600 текстов открыты после выбора:
улучшены {control['improved']} запросов, ухудшены {control['worsened']}. Парный bootstrap,
4000 повторов: 95% интервал изменения Recall@50 [{ci[0]:.6f}; {ci[1]:.6f}].
Интервал включает ноль: статистически устойчивый прирост не установлен.
Замороженные модели v4 могли использовать контрольные взаимодействия во
вспомогательной истории. Для нового ранкера эти тексты и целевые взаимодействия
исключены из обучения; сравнение со старым ансамблем имеет указанное ограничение.
Локальные результаты не гарантируют прирост на закрытой платформе.

## Данные, признаки и обучение

- Полные gold-метки фиксируются из train до очистки вспомогательной истории.
  Единственный фильтр меток — присутствие item в доступном корпусе.
- 19031 обучающий контекст, 8174 нормализованных текста, 21480 положительных пар.
  3400 development-контекстов и 600 контрольных текстов исключены из нового fit.
- OOF для нового текста группирует все его контексты; OOF для известного текста
  удерживает полный контекст и допускает другие контексты этого текста в истории.
  Собственные labels никогда не входят в признаки. Фактическая доступность истории
  измеряется; веса групп согласуют долю известных текстов с benchmark (37.36%).
- 45 признаков: прежние текстовые оценки/ранги/покрытие, география, совместимость
  microcat, цена, рейтинг, отзывы, контакты и E5; дополнительно объём и уверенность
  истории текста, наличие центра, вероятность и поддержка перехода локаций.
  Сырые query_id и item_id не являются входами модели.
- Первый проход: {initial['rows']:,} пар, {initial['mean_group_size']:.1f} кандидата на группу.
  Второй: {mined['rows']:,} пар, {mined['mean_group_size']:.1f} кандидата на группу.
  Добавляем наиболее высоко оценённые собственным пилотным ранкером negatives.
  Отсутствие взаимодействия — слабый минус, а не доказанная нерелевантность.
- Положительные объявления, пропущенные поиском, не добавляются в пул искусственно.
  Recall@50 при выборе всегда считается по полному пулу и полным gold-меткам.
- LambdaRank: 400 деревьев, 31 лист, learning rate 0.05, L2 10, truncation 55,
  label_gain [0,1], seed 260926, 8 потоков, deterministic + force_col_wise.
- Финальная модель обучена на всех {manifest['final_contexts']} доступных размеченных
  контекстах: {final['groups']} групп двух режимов, {final['rows']:,} пар.

## Ошибки и выводы из экспериментов

Исправлена потеря семи меток у шести из прежних 800 контрольных запросов.
Удаление interactions из истории теперь не изменяет gold. Согласование с benchmark
использует только фактически известные тексты в соответствующем срезе.

Широкая выборка и смешанные режимы без дополнительного отбора negatives не
улучшили предыдущую версию. Лучший результат получен при добавлении конкурентов,
высоко оценённых новым пилотом. При одинаковых 400 деревьях и весе 75% результат
mixed_wide — 0.927354, mixed_mined — 0.947600. Эти варианты используют те же
положительные группы и признаки, отличаются отбором negatives.

Отчёты: development_experiments.csv, development_query_deltas.csv,
development_positive_ranks.csv, error_summary.json и feature_importance.csv
в artifacts/ranking-v5. Важность признаков не является доказательством причинного вклада.
География остаётся мягким сигналом: разные location_id не означают гарантированную
нерелевантность. Без timestamps нельзя честно оценить будущие повторы идентичного
контекста; доля таких benchmark-контекстов 4.40% указана отдельно в протоколе.

## Воспроизведение

1. Распаковать deliverables/avito_v5_solution.zip и положить рядом три Parquet задания.
2. Установить requirements.txt; проверенная среда — Python 3.14.6.
3. Открыть Avito.ipynb: Restart Kernel → Run All.

GPU RTX 5070 Ti 12 ГБ использовалась для кодирования отсутствующих E5-векторов.
С приложенными векторами и весами итоговый inference работает на CPU.
{reproduction}
{refit_description}

По умолчанию settings уже выбраны по development: notebook воспроизводит ответ.
AVITO_RUN_MODEL_SEARCH=1 до запуска ядра повторяет полный сравнительный эксперимент.
AVITO_REBUILD_CACHE=1 перестраивает индексы, векторы и финальную модель.
Четыре HGB-priors, два ранкера v4 и собственный пилот для mining поставляются как
замороженные компоненты. Их источники и SHA256 — weights_manifest.json.
Рецептуры: Avito_v0.3.ipynb, Avito_v0.4.ipynb, Avito_training_v5.ipynb.
Большие поисковые кеши и исходные Parquet в архив не включены.

CSV проверен: ровно 2452 query_id, две колонки, по 50 уникальных существующих
item_id, исходный регистр идентификаторов и UTF-8 без индекса.
SHA256 answer.csv: `{manifest['answer_sha256']}`.
Предыдущий ответ с платформенным 0.882692 сохранён как answer_v0.4.csv.

Открытые компоненты: intfloat/multilingual-e5-small (MIT, ревизия
614241f622f53c4eeff9890bdc4f31cfecc418b3), Transformers/PyTorch, LightGBM,
scikit-learn, Snowball, NumPy/SciPy, pandas/PyArrow/joblib.
Модель E5 заморожена; нейросетевого дообучения в v5 не выполнялось.
Чужие решения задания не копировались. План следующих экспериментов — IMPROVEMENT_PLAN.md.
'''
(ROOT/'README.md').write_text(text,encoding='utf-8')
changelog=ROOT/'CHANGELOG.md'
old=changelog.read_text(encoding='utf-8')
if '## v0.5.0 —' not in old:
    entry=f'''## v0.5.0 — полные метки, смешанная история и трудные negatives

- 45 признаков; полные gold-метки отдельно от очищаемой истории.
- 19031 оценочный обучающий контекст; два режима OOF, {mined['rows']:,} пар после mining.
- Выбран LambdaRank 400 деревьев, 75% нового ранкера / 25% v4 по RRF.
- Development: {baseline['matched_recall50']:.6f} → {winner['matched_recall50']:.6f}.
- Контрольные 600: {control['v4_recall50']:.6f} → {control['selected_recall50']:.6f};
  {control['improved']} улучшений / {control['worsened']} ухудшений, bootstrap95 [{ci[0]:.6f}; {ci[1]:.6f}].
- Прирост предварительный; контроль имеет ограничение из-за frozen priors v4.
- Финальное обучение на {manifest['final_contexts']} контекстах / {final['rows']:,} парах.
- Платформенная оценка новой версии пока неизвестна; v4 сохранена как резерв.
- SHA256 ответа: `{manifest['answer_sha256']}`.

'''
    changelog.write_text(old.replace('# История версий\n\n','# История версий\n\n'+entry,1),encoding='utf-8')
print('Written v5 README, changelog and frozen-component checksums.')
