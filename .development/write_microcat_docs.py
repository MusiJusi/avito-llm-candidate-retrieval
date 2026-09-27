"""Document the selected microcat stage and the remaining neural experiments."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
CACHE=ROOT/'artifacts/microcat-v6'
manifest=json.loads((CACHE/'manifest.json').read_text())
selection=manifest['selection'];winner=selection['winner'];baseline=selection['baseline']
control=manifest['control'];classification=json.loads((CACHE/'classification_report.json').read_text())
assert hashlib.sha256((ROOT/manifest['answer_file']).read_bytes()).hexdigest()==manifest['answer_sha256']
v5=ROOT/'artifacts/ranking-v5'
old_weights=json.loads((v5/'weights_manifest.json').read_text())
frozen=dict(old_weights['frozen_components'])
frozen.update({f'artifacts/ranking-v5/{name}':digest for name,digest in old_weights['final_model'].items()})
cached=[CACHE/manifest['final_ranker'],CACHE/manifest['input_vectors']]
cached += [CACHE/name for name in manifest['final_classifiers'].values()]
cached += [v5/name for name in old_weights['embedding_sha256']]
weights={'input_sha256':manifest['input_sha256'],'frozen_components':frozen,
    'cache_components':{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in cached},
    'own_classifier_provenance':'All final classifiers use the provided train; OOF histories exclude own labels and cold items.',
    'e5_encoder_finetuned':False,'v5_components':'Frozen own components, supplied with code; see Avito_v0.5.ipynb.'}
(CACHE/'weights_manifest.json').write_text(json.dumps(weights,indent=2),encoding='utf-8')
portable=CACHE/'portable_reproduction.json'
verification=json.loads(portable.read_text()) if portable.exists() else None
verified=(f"Свежая распаковка архива на CPU без сети создала идентичные байты CSV за {verification['elapsed_seconds']:.2f} с." if verification
          else 'Проверка свежей распаковки архива выполняется после упаковки.')
ce=ROOT/'artifacts/cross-encoder-pilot/pilot_report.json'
ce_result=json.loads(ce.read_text()) if ce.exists() else None
ce_note=(f"Пилот frozen cross-encoder: смешанный Recall@50 на 1000 development-контекстах "
         f"{ce_result['baseline']['mixed_recall50']:.6f} → {ce_result['winner']['mixed_recall50']:.6f}. "
         "Это проверка прямой смеси на shortlist; CE не включён в отправляемую microcat-версию." if ce_result else
         'Пилот cross-encoder готовится отдельно; он не включён в отправляемую microcat-версию.')
text=f'''# Отбор кандидатов услуг Авито: v0.6.0

**Avito_microcat_candidate_v6.ipynb** — самостоятельный notebook, который создаёт **answer_microcat_v6.csv**.
Это экспериментальный кандидат: development улучшился, контроль немного ухудшился.
Основной **answer.csv** и **Avito.ipynb** от v5 сохранены без изменения.
Новый сигнал предсказывает распределение `item_microcat_id` из текста запроса и фильтров,
используя большую часть train, включая выбранные объявления вне benchmark_items.
`search_category` не входит в scoring. Идентификаторы, локация и признаки объявления не входят в текстовые классификаторы.
Последующий ранкер по-прежнему использует разрешённые характеристики объявления и географию.

## Результаты и ограничения

| Проверка Recall@50 | v5 | microcat v6 |
|---|---:|---:|
| Development с согласованием доступности истории и срезов | {baseline['matched_recall50']:.6f} | {winner['matched_recall50']:.6f} |
| Новые тексты, macro | {baseline['unseen_macro_recall50']:.6f} | {winner['unseen_macro_recall50']:.6f} |
| Известные тексты с другим контекстом, macro | {baseline['actual_known_macro_recall50']:.6f} | {winner['actual_known_macro_recall50']:.6f} |
| Контрольные 600 контекстов | {control['v5_recall50']:.6f} | {control['micro_recall50']:.6f} |

Выбран вариант **{winner['variant']}**, {int(winner['trees'])} деревьев,
RRF-вес нового ранкера {winner['weight']:.2f}; оставшаяся доля — замороженная v5.
На контроле {control['improved']} улучшений и {control['worsened']} ухудшений;
парный bootstrap95 для разницы: {control['paired_bootstrap95']}.
Контроль уже оценивался в v5; v4-компоненты имеют известное ограничение по вспомогательной истории.
Он не используется для настройки текущего победителя, но не является новым независимым тестом.
Улучшение на платформе не обещается: известные результаты — v3 0.858569 и v4 0.882692.

## Данные, обучение и ablation

- Полные gold-метки неизменны и отделены от очищаемой истории. Multiple positives учитываются полностью.
- Текстовый microcat-компонент: MultinomialNB на хешированных символьных n-gram 3–5 и словных 1–2;
  запрос и фильтры имеют отдельные блоки признаков. Хеширование не требует обучения словаря на holdout.
- Семантический компонент: MLP 768 → 384 → 212, GELU, dropout 0.15.
  Вход — два замороженных 384-мерных E5-вектора: запрос и фильтры. Пустые фильтры дают нулевой вектор.
  35 эпох AdamW, learning rate 0.0015, weight decay 0.01, batch 512, seed 261827.
- E5 **не дообучалась** в этом этапе. Обучается только классификационная MLP поверх её векторов.
- Разметка query/filter агрегируется в мягкое распределение по всем наблюдённым microcat.
  Частотные входы имеют ограниченный вес sqrt(count), максимум 16; неоднозначные запросы не сводятся к одному классу.
- В train 212 microcat, в корпусе 752. Отсутствующие в fit классы получают NaN в вероятностных признаках
  и явный supported=0. Они не фильтруются и не превращаются в искусственно доказанный минус.
- Для каждого из 3 OOF-фолдов в двух режимах классификатор обучается на разрешённой истории:
  исключены собственные контексты, для новых текстов — весь текст; применено то же удаление cold items, что в v5.
  И development, и контроль исключены из истории обучения оценочного ранкера.
- Оценочные классификаторы обучались на 379914 парах для новых текстов и 456891 для известных.
  На новых текстах microcat Recall@5: NB {classification['classifiers']['unseen_text']['nb']['micro_recall5']:.6f},
  MLP {classification['classifiers']['unseen_text']['mlp']['micro_recall5']:.6f}. Это другая метрика, не Recall@50 объявлений.
- Для каждого кандидата добавлены вероятность microcat, log probability, log rank,
  peak, entropy, margin и supported. Отдельно проверены NB, MLP и оба набора признаков.
- Во всех ablation одинаковые 14,222,085 пар, положительные метки, порядок кандидатов,
  веса групп и гиперпараметры LambdaRank. Проверки равенства сопоставляют каждую группу с v5.
  Меняется только набор новых признаков, затем выбирается RRF-смесь на development.
- Итоговый классификатор использует все {manifest['classifier_training_pairs']:,} разрешённых пар train.
  Финальный ранкер обучается на {manifest['final_contexts']:,} контекстах в двух OOF-режимах.
  Число входных признаков выбранного ранкера — {manifest['feature_count']}.

## Анализ ошибок и следующие этапы

Ошибочный top-1 microcat остаётся частым, поэтому жёсткие ограничения по прогнозируемому виду услуги не применяются.
NB и MLP сравниваются отдельно; наличие нового классификатора само по себе не считается улучшением поиска.
Новый компонент принимается по Recall@50 на полном исходном пуле и по полным положительным меткам.
Все результаты ablation и OOF-аудиты сохранены в artifacts/microcat-v6.

{ce_note}
Microcat-версия сохраняет исходную E5. Отдельно уже выполнен пилот дообучения query-encoder
и диагностика cold items; география по типу услуги и OOF-оценки learned E5 исследуются отдельно.
Текущий статус и численные результаты — в [IMPROVEMENT_PROGRESS.md](IMPROVEMENT_PROGRESS.md)
и `Avito_neural_experiments.ipynb`. Обучаемая интеграция CE пока не выполнена.
Этот microcat-этап не закрывает весь IMPROVEMENT_PLAN.md.

## Локальное воспроизведение

1. Распаковать deliverables/avito_v6_experiments.zip; положить рядом три Parquet из задания.
2. Установить requirements.txt (проверенная среда Python 3.14.6).
3. Открыть Avito_microcat_candidate_v6.ipynb: Restart Kernel → Run All.

Нормальный запуск использует поставляемые веса и E5-векторы и работает на CPU без inference API.
{verified}
AVITO_RUN_MODEL_SEARCH=1 повторяет исследование; AVITO_REBUILD_CACHE=1 перестраивает новые индексы,
векторы и финальный microcat-ранкер. Собственные компоненты v5 остаются замороженными и поставляются
с весами: исходные рецептуры в Avito_v0.3.ipynb, Avito_v0.4.ipynb, Avito_v0.5.ipynb.
Большие OOF-пулы и sampled datasets не входят в архив; при полном переобучении строятся из Parquet.
Контрольные суммы компонентов и исходных данных — artifacts/microcat-v6/weights_manifest.json.
Avito_microcat_v6.ipynb содержит самостоятельное воспроизведение исследовательского этапа.

Формат CSV: 2452 query_id, две колонки, 50 уникальных существующих item_id на строку, исходный регистр,
UTF-8, пробел между IDs, окончания строк LF. SHA256: `{manifest['answer_sha256']}`.
Предыдущий ответ сохранён в answer_v0.5.csv, более ранний — answer_v0.4.csv.

Открытые компоненты: intfloat/multilingual-e5-small (MIT, ревизия 614241f622f53c4eeff9890bdc4f31cfecc418b3),
PyTorch/Transformers, LightGBM, scikit-learn, NumPy/SciPy, pandas/PyArrow, joblib, Snowball.
Собственные обученные модели используют только предоставленные данные.
Документация [MultinomialNB](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html)
и [HashingVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.HashingVectorizer.html).
'''
previous=ROOT/'README_v0.5.md'
if not previous.exists():previous.write_bytes((ROOT/'README.md').read_bytes())
(ROOT/'README_microcat_v6.md').write_text(text,encoding='utf-8')
change=ROOT/'CHANGELOG.md'
old=change.read_text(encoding='utf-8')
if '## v0.6.0' not in old:
    entry=f'''## v0.6.0 — microcat из большей части train

- NB и MLP на frozen E5, мягкие множественные цели, строгие OOF-признаки.
- Три отдельных feature ablation на неизменных mined-группах v5.
- Development {baseline['matched_recall50']:.6f} → {winner['matched_recall50']:.6f}.
- Контроль {control['v5_recall50']:.6f} → {control['micro_recall50']:.6f}; ограничение независимости явно сохранено.
- Переобучение на полном разрешённом train; отдельный экспериментальный CSV и notebook, основной ответ v5 не меняется.
- E5 query-encoder в этом этапе не дообучалась; CE не включён в отправляемую microcat-версию.
- SHA256 answer_microcat_v6.csv: `{manifest['answer_sha256']}`.

'''
    change.write_text(old.replace('# История версий\n\n','# История версий\n\n'+entry,1),encoding='utf-8')
print('Written microcat documentation and weight hashes.')
