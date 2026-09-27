"""Summarize measured v9 artifacts, distinguishing progress from final quality."""
from pathlib import Path
import csv
import json
ROOT=Path(__file__).resolve().parents[1]
CACHE=ROOT/'artifacts/context-v9'
def read(name):
    path=CACHE/name
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else None
def report():
    selection=read('selection.json');control=read('control.json');manifest=read('manifest.json');proof=read('reproduction.json')
    data={'selection':selection,'control':control,'manifest':manifest,'reproduction':proof,
        'training_reports':[json.loads(p.read_text()) for p in CACHE.glob('*_training_*.json')]}
    (CACHE/'progress.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
    lines=['# Независимые эксперименты с контекстом v9\n',
        'Ветка: `feature/context-v9`. Подтверждённый v7 (платформа 0.897443) и отдельный v8 сохраняются. '
        'Исследовательский запуск — `experiments/Avito_neural_experiments.ipynb`.\n',
        'Используются общие идеи географического восстановления и kNN-регрессии. '
        'Публичное сравнение подходов: https://github.com/nikitabaskov/avito_ds_bootcamp, '
        'состояние 05b8d49bd3d8145d3894fff1321bc26e4adae6cc. Чужой код, веса, ответы '
        'и конкретные конфигурации в реализации не используются.\n',
        '## Что проверяем\n',
        '- Исторический центр при отсутствии центра в корпусе; источник координат, объём истории, '
        'медианный и хвостовой разброс, непрерывная уверенность, распределение переходов.\n'
        '- Семантические соседи: распределение микрокатегорий, усреднённый вектор выбранных '
        'объявлений, уверенность и покрытие. Используются готовые замороженные E5-векторы.\n'
        '- Расширенный пул learned E5 плюс отдельные географические кандидаты. '
        'Более широкие обучающие группы с трудными конкурентами и случайным хвостом.\n',
        'Признаки считаются из очищенной истории, с исключением собственных контекстов и '
        'контролем history-digest шести OOF-encoder. `search_category` не передаётся scoring.\n',
        '## Данные и инженерные проверки\n',
        'Координаты в train представлены строками: явно преобразуем их в float, '
        'некорректные значения, (0, 0) и значения вне географических границ заменяем '
        'на отсутствующие. Центры корпуса строятся без разметки; исторические центры, '
        'разброс и переходы — только из разрешённых взаимодействий. '
        'Тест исключения held-меток проверяет, что координаты и микрокатегория '
        'удержанного объявления не влияют на новые агрегаты.\n',
        'Семантическое соседство использует 24 ближайших текста исходной E5 и мягкие '
        'веса с температурой 0.1. Усредняются распределения микрокатегорий и frozen '
        'векторы выбранных объявлений, включая объявления train вне корпуса. '
        'Дополнительно сохраняются покрытие и уверенность, поэтому редкая или '
        'неоднозначная история не превращается в жёсткий фильтр.\n',
        '## Измерения\n']
    pilot=CACHE/'pilot/development.csv'
    if pilot.exists():
        with pilot.open(newline='',encoding='utf-8') as stream:pilot_rows=list(csv.DictReader(stream))
        lines.append('Первый пилот: 4000 контекстов, 7985 групп и 5 951 194 пары. '
            'Ни один новый ранкер не превзошёл v8. Наиболее близким оказался вариант '
            'с географией (вес 0.5): matched 0.951292 против 0.952815. '
            'Просадка варианта без новых признаков указывает на ограничение малого '
            'обучающего набора, но сама по себе не доказывает его причину.\n')
    table=CACHE/'development.csv'
    if table.exists():
        with table.open(newline='',encoding='utf-8') as stream:rows=list(csv.DictReader(stream))
        lines.append('| Вариант | Вес | Matched Recall@50 | Новые тексты | Новый контекст |\n'
                     '| --- | ---: | ---: | ---: | ---: |')
        for row in rows:
            lines.append(f'| {row["variant"]} | {row["weight"]} | '
                f'{float(row["matched_recall50"]):.6f} | {float(row["unseen_macro"]):.6f} | '
                f'{float(row["held_macro"]):.6f} |')
        lines.append('')
    for training in data['training_reports']:
        lines.append(f'Обучающие группы {training["stage"]}: {training["groups"]}, '
            f'строк {training["rows"]:,}, в среднем {training["mean_group_size"]:.1f} кандидата.\n')
    if selection:
        winner=selection['winner']
        lines.append(f'Выбран по development: **{winner["variant"]}**, вес {winner["weight"]}, '
            f'matched Recall@50 **{winner["matched_recall50"]:.6f}**; v8 **0.952815**.\n')
    else:lines.append('Выбор конфигурации ещё не завершён; новых измеренных итоговых метрик пока нет.\n')
    if control:lines.append(f'Ранее просмотренный контроль: v8 {control["v8_recall50"]:.6f}, '
        f'выбранный вариант {control["selected_recall50"]:.6f}. Не использовался для выбора конфигурации.\n')
    if manifest:lines.append(f'Ответ: `{manifest["answer_file"]}`, SHA-256 `{manifest["answer_sha256"]}`.\n')
    if proof:lines.append(f'CPU без сети: {proof["elapsed_seconds"]:.2f} секунд, побайтовое совпадение CSV.\n')
    errors=read('error_analysis.json')
    if errors:
        lines.append('## Анализ ошибок выбранной конфигурации\n')
        lines.append(f'На контроле улучшены {errors["improved"]} запросов, ухудшены '
            f'{errors["worsened"]}, без изменений {errors["unchanged"]}. '
            'Разбор выполнен после выбора модели; конфигурация по этим ошибкам не менялась. '
            'Полная таблица: `artifacts/context-v9/control_errors.csv`.\n')
        for kind,title in [('lost_examples','Примеры ухудшений'),('gained_examples','Примеры улучшений')]:
            lines.append(title+':\n')
            for row in errors[kind][:4]:
                lines.append(f'- «{row["search_query"]}»: {row["incumbent_recall50"]:.2f} → '
                    f'{row["selected_recall50"]:.2f}; источник географии {row["geography_source"]:.0f}, '
                    f'уверенность {row["geography_confidence"]:.3f}.')
            lines.append('')
        lines.append('Источники географии: 0 — центр корпуса, 1 — историческое восстановление, '
            '2 — координат нет. Метрика контроля ухудшилась: выигрыш development '
            'остаётся небольшим и не гарантирует улучшения платформы.\n')
    cat=ROOT/'artifacts/context-catboost-v9/selection.json'
    if cat.exists():
        result=json.loads(cat.read_text())['winner']
        lines.append(f'Отдельный пилот CatBoost YetiRank (400 деревьев, глубина 6, '
            f'первые 4000 контекстов): выбран {result["variant"]}, '
            f'matched Recall@50 {result["matched_recall50"]:.6f}. '
            'GPU-обучение не заявляется побайтово детерминированным; '
            'воспроизводимость ответа проверяется с зафиксированными весами.\n')
    lines+=['## Ограничения\n',
        'Development и контроль ранее просматривались, старые v4 priors имеют известное '
        'ограничение истории. Первый пилот обучается на подвыборке контекстов. '
        'Mining выполняет ранее обученный v7-ранкер; это обычный training-mining, '
        'но не независимый OOF-miner. Невыбранные объявления остаются неразмеченными. '
        'Точность на платформе нельзя вывести из локальной оценки.\n']
    (ROOT/'docs/EXPERIMENTS_V9.md').write_text('\n'.join(lines).rstrip()+'\n',encoding='utf-8')
    print('Context v9 report updated.')
if __name__=='__main__':report()
