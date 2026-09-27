"""Document actual outcomes, distinguishing pilots from final integration."""
from pathlib import Path
import json
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
def read(folder,name):
    path=ROOT/'artifacts'/folder/name
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else None
v5=read('ranking-v5','manifest.json')
micro=read('microcat-v6','manifest.json')
qe=read('query-encoder-pilot','pilot_report.json')
qc=read('query-encoder-pilot','control_report.json')
qr=read('query-encoder-rank','selection.json')
qrc=read('query-encoder-rank','control.json')
geo=read('service-geography-v6','selection.json')
geoc=read('service-geography-v6','control.json')
blend=read('query-micro-blend','report.json')
final=read('query-encoder-candidate','manifest.json')
portable=read('query-encoder-candidate','portable_reproduction.json')
rows=[f"| v5 | {v5['selection']['winner']['matched_recall50']:.6f} | {v5['control']['selected_recall50']:.6f} | Основной ответ |",
      f"| microcat NB+MLP | {micro['selection']['winner']['matched_recall50']:.6f} | {micro['control']['micro_recall50']:.6f} | Отдельный экспериментальный CSV |"]
if qe:rows.append(f"| E5 query-encoder, прямая смесь | {qe['winner']['matched_recall50']:.6f} | {qc['query_encoder_recall50']:.6f} | Пилот, фиксированный пул v5 |" if qc else f"| E5 query-encoder | {qe['winner']['matched_recall50']:.6f} | Не проверен | Пилот |")
if qr:rows.append(f"| E5: отбор после OOF-обучения | {qr['winner']['matched_recall50']:.6f} | {qrc['selected_recall50']:.6f} | Победитель: {qr['winner']['variant']} |" if qrc else '| E5: OOF-ранкер | Выбран | Контроль выполняется | Эксперимент |')
if geo:rows.append(f"| География по виду услуги | {geo['winner']['matched_recall50']:.6f} | {geoc['selected_recall50']:.6f} | Победитель: {geo['winner']['variant']} |" if geoc else '| География | Выбрана | Контроль выполняется | Эксперимент |')
text='''# Фактический прогресс по плану улучшений

Это дополнение к исходному `IMPROVEMENT_PLAN.md`, который описывает отправную
точку v4. Основной `answer.csv` пока соответствует воспроизводимой v5.
Метрика платформы для v4 — 0.882692; для новых вариантов она пока неизвестна.
`search_category` не используется в scoring.

| Вариант | Development Recall@50, matched | Контроль, 600 контекстов | Статус |
| --- | ---: | ---: | --- |
'''+ '\n'.join(rows)+'''

## 1. Валидация

Выполнено: неизменяемые gold из полного train, два режима истории (новый текст
и известный текст с удержанным контекстом), проверки отсутствия собственных
контекстов в истории, исправление семи потерянных положительных меток.
Новые 4000 контекстов разделены на development 3400 и контроль 600.
Контроль уже просмотрен в предыдущих итерациях и не считается новым независимым
тестом. У замороженных priors v4 есть ограничение по использованной вспомогательной
истории, явно отмеченное в отчётах.

Cold-item диагностика выполнена на фиксированных 400 development-контекстах:
при удалении 0/50/90/100% собственных положительных items Recall@50 v5 остался
0.9275 в обоих режимах; полнота пула 0.9875. Доля видимых положительных items
действительно менялась. Это ограниченная диагностика, а не доказательство
нечувствительности скрытого бенчмарка. Не выполнен полный перебор обучения
ранкера под разные доли новых items.

## 2. Ранкер и география

Выполнено в v5: широкие группы, трудные отрицательные примеры из собственного
mixed-wide ранкера, признаки уверенности истории, отдельные признаки доступности
центра, переходов между локациями и объёма наблюдений. Обучение использует оба
режима OOF; 14.2 млн пар при выборе и 19.8 млн при финальном refit.
География по виду услуги реализована отдельно: пять сглаженных признаков, шесть
изолированных историй OOF, две ablation (geo и microcat+geo). Результаты —
`artifacts/service-geography-v6`. Жёсткого географического фильтра нет.

## 3. Большая часть train и E5

Microcat: два классификатора NB/MLP, мягкие множественные цели, 14 новых признаков,
шесть изолированных OOF-fit; неизвестные классы остаются допустимыми кандидатами.
Финальные классификаторы используют 467054 уникальных разрешённых пар train.
Development вырос, контроль снизился; основной ответ автоматически не заменён.

E5 query-encoder: пилот обучен на 379988 разрешённых парах (69710 текстов,
209130 выбранных пар за три эпохи). Document-encoder и 292210 document-векторов
заморожены; использованы в том числе документы вне benchmark_items. Исходные
веса E5 проверены по SHA256. Память обучения около 4 ГБ на RTX 5070 Ti.
Многоположительный contrastive loss и регуляризация к исходным query-векторам.
Результат прямой смеси проверен на контроле с уже выбранными параметрами.
Измерялся отбор внутри существующего пула v5; улучшение глобальной полноты
нового пула из этого пилота не следует.

Для обучения ранкера добавлен отдельный эксперимент из шести OOF query-encoder.
Группы и отрицательные примеры равны v5; признаки включают learned cosine,
географический score, изменения к frozen E5 и ранги внутри полного пула.
Проверка equality сопоставляет все sampled группы с исходной матрицей v5.

## 4. Cross-encoder

Выполнен frozen-пилот `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` на 1000
development-контекстах: shortlist 300/500, 500000 пар, около 687 секунд,
пиковая память модели около 484 МиБ. Прямая смесь не улучшила Recall@50;
лучший вес CE равен нулю. Это не проверка CE как обучаемого признака ранкера
и не доказательство отсутствия пользы дообучения CE. Эти две части плана
пока не выполнены. CE не включён в текущие ответы.

## 5. Воспроизводимость и артефакты

`Avito.ipynb` и основной ответ v5 прошли offline/CPU-воспроизведение.
`Avito_microcat_candidate_v6.ipynb` отдельно воспроизводит
`answer_microcat_v6.csv`: свежая распаковка, CPU, сеть отключена, байты равны,
около 303 секунд. SHA256 и полная проверка CSV сохранены в manifest/validation.
Самостоятельный исследовательский notebook — `Avito_microcat_v6.ipynb`.
Jupyter-запуск новых пилотов — `Avito_neural_experiments.ipynb`, с комментированными
локальными исходниками в `.development`. По умолчанию показывает фактические
отчёты; переключатели `RUN_*` повторяют обучение.

Старые версии и новые кандидаты сохраняются отдельно; ни один отрицательный
эксперимент не подменяет основной CSV. Полный исходный план ещё не закрыт:
остаётся интеграция полезных сигналов, финальный neural refit и его воспроизведение,
а также условные CE-эксперименты, если для них будет достаточно времени.
'''
if blend:
    text+='\n## Проверка ансамбля E5 + microcat\n\n'
    text+=f"Development: {blend['baseline']['matched_recall50']:.6f} → {blend['winner']['matched_recall50']:.6f}; выбран вес microcat {blend['winner']['micro_weight']}.\n"
    if blend['control']:
        text+=f"Контроль: {blend['control']['query_only']:.6f} → {blend['control']['blend']:.6f}; {blend['control']['improved']} улучшений, {blend['control']['worsened']} ухудшений. Ансамбль не включён в отправляемый neural-кандидат.\n"
if final:
    text=text.replace('остаётся интеграция полезных сигналов, финальный neural refit и его воспроизведение,\nа также условные CE-эксперименты, если для них будет достаточно времени.',
        'не выполнены обучаемый CE-признак и условное дообучение CE, а также полное повторное обучение под разные cold-item доли.')
    text+='\n## Финальный neural-кандидат\n\n'
    text+=f"Полный refit выполнен: {final['training_pairs']} доступных пар для query-encoder, 221619 выбранных примеров за три эпохи с возможными повторами; 51 признак для выбранного ранкера. CSV: `{final['answer_file']}`, SHA256 `{final['answer_sha256']}`.\n"
    if portable:text+=f"Свежая распаковка CPU/offline создала побайтово идентичный CSV за {portable['elapsed_seconds']:.2f} секунд. Корневой v5-ответ сохранён.\n"
    else:text+='CPU/offline-проверка свежей распаковки выполняется отдельно.\n'
(ROOT/'IMPROVEMENT_PROGRESS.md').write_text(text,encoding='utf-8')
(ROOT/'artifacts/improvement_progress.json').write_text(json.dumps({'updated_utc':datetime.now(timezone.utc).isoformat(),
    'primary':'v5','recommended_next_submission':'query-encoder-candidate' if final else None,
    'microcat':'experimental_candidate','query_encoder_pilot':'completed',
    'query_ranker':'completed' if qrc else 'running','service_geo':'completed' if geoc else 'running',
    'neural_final_refit':'completed' if final else 'running','neural_cpu_offline_reproduced':bool(portable),
    'ce_trained_feature_completed':False,'whole_plan_completed':False},indent=2),encoding='utf-8')
print('Updated factual progress report.')
