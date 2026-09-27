"""Describe the actual final neural candidate and its reproducibility evidence."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]
CACHE=ROOT/'artifacts/query-encoder-candidate'
manifest=json.loads((CACHE/'manifest.json').read_text())
submission=ROOT/'deliverables/query_encoder_submission/answer.csv'
submission.parent.mkdir(parents=True,exist_ok=True)
submission.write_bytes((ROOT/manifest['answer_file']).read_bytes())
winner=manifest['selection']['winner'];control=manifest['control']
replay=CACHE/'portable_reproduction.json'
reproduction=json.loads(replay.read_text()) if replay.exists() else None
errors_path=ROOT/'artifacts/query-encoder-rank/error_summary.json'
errors=json.loads(errors_path.read_text()) if errors_path.exists() else None
error_note=''
if errors:
    unseen=errors['unseen_text']
    error_note=f"На новых текстах из {unseen['missed_pairs']} пропущенных положительных пар {unseen['outside_pool']} вне пула, {unseen['ranks51_100']} на позициях 51–100 и {unseen['ranks101_300']} на 101–300. Доля найденных пар при совпадающей локации {unseen['same_location_pair_hit50']:.3%}, при разных {unseen['different_location_pair_hit50']:.3%}; это не макро-Recall по запросам. Положительные ранги, изменения по запросам и gain признаков сохранены в artifacts/query-encoder-rank."
evidence=(f"Свежая распаковка на CPU с отключённой сетью и удалённым CSV получила идентичные байты за {reproduction['elapsed_seconds']:.2f} секунд."
    if reproduction else 'Свежая CPU/offline-проверка архива ещё не завершена.')
text=f'''# Отбор кандидатов с дообученным query-encoder E5

`Avito_query_encoder_candidate.ipynb` создаёт **answer_query_encoder.csv**.
Идентичная копия с именем для отправки — `deliverables/query_encoder_submission/answer.csv`.
Основной `answer.csv` v5 сохранён отдельно. Новый файл — самостоятельный кандидат
для следующей отправки; метрика платформы для него ещё неизвестна.
Выбранный на development вариант: **{winner['variant']}**, вес {winner['weight']}.

| Локальная проверка Recall@50 | v5 | Neural-кандидат |
| --- | ---: | ---: |
| Development, matched по признакам запросов | {manifest['selection']['baseline']['matched_recall50']:.6f} | {winner['matched_recall50']:.6f} |
| Контроль, 600 контекстов | {control['v5_recall50']:.6f} | {control['selected_recall50']:.6f} |

На контроле улучшились {control['improved']} запросов, ухудшились {control['worsened']}.
Парный bootstrap 95% для изменения: {control['bootstrap95']}.
Контроль ранее просмотрен в нескольких экспериментах и не является новым независимым тестом.
Замороженные вспомогательные priors v4 имеют ограничение по использованной истории.
Локальное улучшение не гарантирует прирост на скрытой платформе.

## Данные и модели

Исходный `intfloat/multilingual-e5-small` кодирует документы корпуса и документы train.
Query-encoder — отдельная копия этой модели, дообученная на разрешённых положительных
парах. Document-encoder заморожен, готовые корпусные векторы сохраняются. Для пилота
доступны 379988 пар, включая объявления вне benchmark_items; для финального encoder
доступны {manifest['training_pairs']} уникальных пар полного train. За три эпохи
пилот выбирает 209130 пар; финальное обучение выбирает 221619 пар, с возможными
повторами. На каждый уникальный текст в каждой эпохе выбирается один из его
известных положительных документов, поэтому частые запросы не доминируют в loss.

Query-encoder получает нормализованный текст запроса, без `search_category`,
локации, `query_id` и item ID. Фильтры и характеристики исполнителя использует
существующий v5-ранкер. Encoder обучается три эпохи, AdamW 2e-5, batch 128,
temperature 0.05, weight decay 0.01, query max length 64. Minibatch содержит
уникальные тексты, известные положительные документы учитываются многоположительной
маской. Штраф 0.2 к исходному query-вектору ограничивает дрейф.

Кандидатогенерация и документные векторы остаются v5. Проверялись cosine и
географический score дообученной E5, прямые RRF-смеси, затем LambdaRank на шести
OOF-сигналах. Три фолда для новых текстов и три для удержанных контекстов исключают
собственные положительные пары из обучения feature-encoder. Группы и отрицательные
примеры сверены с v5; финальные encoder/ранкер обучаются только после выбора
конфигурации на development. Итоговая конфигурация описана в manifest.

Новые географические признаки по виду услуги не дали дополнительного прироста.
Frozen cross-encoder также не помог при прямом смешивании; он не входит в ответ.
Microcat-классификаторы дали неоднозначную проверку и сохранены отдельным кандидатом.
Полный фактический журнал — `IMPROVEMENT_PROGRESS.md`.

{error_note}

## Проверка и воспроизведение

1. Распаковать `deliverables/avito_query_encoder_solution.zip`.
2. Положить рядом три исходных Parquet и установить `requirements.txt`.
3. Открыть `Avito_query_encoder_candidate.ipynb`: Restart Kernel → Run All.

Обычный запуск работает на CPU без сети: в архиве есть исходная E5, собственные
веса, document/query-векторы и версии старых компонентов. Query-векторы получены
инференсом обученной модели, а не из разметки бенчмарка. Notebook проверяет hashes
весов, исходных файлов и итогового CSV. {evidence}
Большие OOF-пулы в архив не входят. Рецептура обучения и ablation запускается из
`Avito_neural_experiments.ipynb`; полный refit —
`.development/export_query_encoder_candidate.py`. Эти исходники сопровождаются
комментариями и журналами исключения меток.

CSV проверен независимо: ровно 2452 query_id, две колонки, ровно 50 уникальных
существующих item_id на запрос, исходный регистр, UTF-8, один пробел между IDs.
SHA256: `{manifest['answer_sha256']}`.

Open-source: multilingual-e5-small (MIT, ревизия
614241f622f53c4eeff9890bdc4f31cfecc418b3), PyTorch, Transformers, LightGBM,
scikit-learn, NumPy/SciPy, pandas/PyArrow, joblib, Snowball. Собственные модели
используют только предоставленный train. Обращений к внешним inference API нет.
'''
(ROOT/'README_query_encoder.md').write_text(text,encoding='utf-8')
main=ROOT/'README.md'
previous=main.read_text(encoding='utf-8')
marker='<!-- learned-query-candidate -->'
if marker not in previous:
    note='''<!-- learned-query-candidate -->
**Новый кандидат с дообученной E5:** [README_query_encoder.md](README_query_encoder.md).
Его код — `Avito_query_encoder_candidate.ipynb`; файл для отправки —
`deliverables/query_encoder_submission/answer.csv` (также `answer_query_encoder.csv`).
Архив с весами — `deliverables/avito_query_encoder_solution.zip`.
Ниже сохранено описание основной v5: её `Avito.ipynb` и корневой `answer.csv`
оставлены резервными. Для neural-ответа воспроизводите именно новый notebook.

'''
    main.write_text(note+previous,encoding='utf-8')
print('Written learned-query candidate documentation.')
change=ROOT/'CHANGELOG.md'
history=change.read_text(encoding='utf-8')
if '## v0.7.0' not in history:
    entry=f'''## v0.7.0 — query-encoder E5 и OOF-ранкер

- Document-encoder заморожен; отдельный query-encoder обучен на полном train.
- Шесть OOF-encoder исключают собственные метки, 51 признак LambdaRank.
- Development {manifest['selection']['baseline']['matched_recall50']:.6f} → {winner['matched_recall50']:.6f}.
- Контроль {control['v5_recall50']:.6f} → {control['selected_recall50']:.6f}; {control['improved']} улучшений / {control['worsened']} ухудшений.
- География по услуге и ансамбль microcat не дали согласованной пользы; сохранены как эксперименты.
- Отдельный answer_query_encoder.csv; основной v5 остаётся резервом.
- SHA256 answer_query_encoder.csv: `{manifest['answer_sha256']}`.

'''
    change.write_text(history.replace('# История версий\n\n','# История версий\n\n'+entry,1),encoding='utf-8')
