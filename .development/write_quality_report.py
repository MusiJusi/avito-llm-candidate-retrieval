"""Maintain measured v10 results and one navigation entry in the research NB."""
from pathlib import Path
import json
import csv

ROOT=Path(__file__).resolve().parents[1]
CACHE=ROOT/'artifacts/quality-v10'


def read(path):return json.loads(path.read_text(encoding='utf-8')) if path.exists() else None


def table(path):
    if not path.exists():return 'Эксперимент ещё не завершён.\n'
    with path.open(encoding='utf-8',newline='') as stream:rows=list(csv.DictReader(stream))
    columns=list(rows[0])
    lines=['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']
    for row in rows:
        lines.append('| '+' | '.join(f'{float(row[c]):.6f}' if c.endswith('macro') or c.endswith('recall50') else row[c] for c in columns)+' |')
    return '\n'.join(lines)+'\n'


def report():
    selection=read(CACHE/'selection.json');control=read(CACHE/'control.json')
    manifest=read(CACHE/'manifest.json');proof=read(CACHE/'reproduction.json')
    lines=['# Эксперименты v10\n',
        'Ветка `feature/quality-v10`. Лучший подтверждённый результат платформы: '
        '**v9, Recall@50 = 0.900753**. Он сохранён в `experiments/results/v9/answer.csv`. '
        'Для v10 результат платформы пока неизвестен.\n',
        '## Изменения\n',
        '- Пользователь разрешил использовать `search_category`. Сначала проверен мягкий приоритет; '
        'категория 0 не задаёт ограничения. В новом ранкере — совпадение категории и признак объявления услуги.\n'
        '- Используются все 19 031 обучающих контекста development-протокола; при финальном обучении '
        'доступны 26 556 контекстов с положительными объявлениями в корпусе. Оба режима истории сохранены.\n'
        '- Сохранены широкие кандидаты v9. Добавлены конкуренты, выбранные отдельными моделями '
        'с исключением меток текущей группы текстов. Эти модели используют только лексические, '
        'метаданные и исходную замороженную E5. Пулы кандидатов исторические, поэтому это не '
        'заявляется как полностью вложенный OOF-протокол. Обычный mining v7 также остаётся.\n'
        '- Замороженная E5 кодирует запрос вместе с текстовыми фильтрами. Добавлены сходство, '
        'разница с прежним сходством и ранг. Сам encoder не дообучается в этом эксперименте.\n'
        '- История объявления: число взаимодействий, разных текстов запросов и взаимодействий '
        'из искомой локации. Переходы между локациями учитывают вид услуги.\n'
        '- Большие обучающие матрицы собираются на диске по частям. Вариант покрытия и варианты '
        'с новыми признаками используют одинаковые обучающие примеры.\n',
        '## Категория\n',table(CACHE/'category_development.csv'),
        '## Ранкер\n',table(CACHE/'development.csv')]
    if selection:lines.append('Выбор по development, до просмотра контроля:\n\n```json\n'+json.dumps(selection['winner'],indent=2)+'\n```\n')
    if control:lines.append('Контроль ранее просматривался, для выбора конфигурации не используется:\n\n```json\n'+json.dumps(control,indent=2)+'\n```\n')
    lines.extend(['## Популярность и доступность истории\n',
        'Отдельная абляция исключает три признака популярности объявления, сохраняя остальные признаки и примеры. '
        'Смешанное обучение добавляет группы с удалением 0% и 50% положительных items из разрешённой истории. '
        'Исходные группы удаляют 90%; история encoder остаётся строгим подмножеством расширенной истории признаков. '
        'Свои тексты/контексты исключены. Положительные метки не удаляются. В смешанной модели доступны четыре '
        'сигнала повторных взаимодействий query-item из train и dropout категории с частотой незаданной категории benchmark.\n',
        table(CACHE/'no_popularity_development.csv'),table(ROOT/'artifacts/warm-history-v10/development.csv'),
        table(CACHE/'warm_deeper_development.csv'),table(CACHE/'warm_aux_development.csv')])
    warm_final=read(CACHE/'warm_aux_selection.json')
    if warm_final and warm_final.get('control_veto_applied'):
        lines.append('Отдельный тёплый ранкер выиграл менее одного попадания на 3400 запросов development, '
            'но ухудшил контроль. Поэтому он исключён из первой отправки. Решение принято после '
            'просмотра контроля: итоговое значение контроля уже нельзя считать независимой '
            'оценкой выбранного состава. Исходный победитель development сохранён в '
            '`warm_aux_choice_before_control.json`, исходное падение — в `warm_aux_control.json`.\n')
    stress=read(CACHE/'category_shift_stress.json')
    if stress:lines.extend(['## Незаданная категория\n',
        'Категория 0 означает отсутствие ограничения. Дополнительно проверено искусственное обнуление категории '
        'у всех validation-запросов; это стресс-проверка, а не новая независимая разметка.\n',
        '```json\n'+json.dumps(stress,indent=2)+'\n```\n'])
    lines.extend(['## Чувствительность к истории\n',
        'При фиксированных весах и query-encoder заново вычисляются разрешённая история, кандидаты и признаки '
        'для долей удаления положительных items 0%, 50%, 90% и 100%. Это проверка доступности истории при inference; '
        'она не доказывает устройство скрытого теста. Сравнение относится к основному ранкеру до дополнительных '
        'моделей полей/BGE.\n',table(CACHE/'history_sensitivity.csv')])
    for folder,title in [('semantic-fields-v10','Представление объявления'),
        ('field-ranker-v10','Признаки полей объявления в ранкере'),('cross-finetune-v10','Дообучение cross-encoder'),
        ('bge-m3-v10','Независимый encoder BGE-M3')]:
        cache=ROOT/'artifacts'/folder
        lines.extend(['## '+title+'\n',table(cache/'development.csv')])
        measured=read(cache/'selection.json')
        if measured:lines.append('Выбран по development: `'+json.dumps(measured['winner'],ensure_ascii=False)+'`.\n')
        measured_control=read(cache/'control.json')
        if measured_control:lines.append('Контроль: `'+json.dumps(measured_control)+'`.\n')
    if manifest:
        lines.append('## Отправка и воспроизведение\n')
        lines.append(f"Ответ: `{manifest['answer_file']}`. SHA-256: `{manifest['answer_sha256']}`. "
            'Notebook: `experiments/Avito_v10_candidate.ipynb`; архив с локальными весами: '
            '`deliverables/avito_v10_solution.zip`.\n')
    if proof:lines.append(f"Свежий запуск {proof['device']} без сети и готового CSV: {proof['elapsed_seconds']:.2f} с, "
        f"{proof['code_cells_executed']} ячеек, побайтовое совпадение ответа.\n")
    lines.extend(['## Повторение экспериментов\n',
        'Все команды выполняются из корня проекта. Основной notebook воспроизводит inference; '
        'следующие команды продолжают обучение из исследовательского workspace v9 и явно запускают '
        'эксперименты на локальных данных. Для обучения с нуля сначала нужны базовые компоненты v5/v7 '
        'и шесть OOF query-encoder, описанные в предыдущих отчётах и в `query_encoder_rank.py`; '
        '`context_scale_v9.py` восстанавливает обучающие пулы v9. Эти крупные обучающие кэши '
        'не включены в архив inference. Большие промежуточные матрицы '
        'не требуются для обычного Run All.\n',
        '```text\npython .development/quality_v10.py\npython .development/train_quality_v10.py\n'
        'python .development/no_popularity_v10.py\npython .development/warm_history_v10.py\n'
        'python .development/warm_deeper_v10.py\npython .development/check_category_shift_v10.py\n'
        'python .development/semantic_fields_v10.py\npython .development/field_ranker_v10.py\n'
        'python .development/history_sensitivity_v10.py\npython .development/prepare_bge_m3_v10.py\n'
        'python .development/encode_bge_m3_v10.py\npython .development/bge_ranker_v10.py\n'
        'python .development/finetune_cross_encoder_v10.py\n'
        'python .development/export_quality_v10.py\npython .development/validate_quality_v10.py\n'
        'python .development/package_quality_v10.py\npython .development/verify_quality_v10.py\n```\n',
        'BGE-M3 — `BAAI/bge-m3`, MIT, фиксированная ревизия '
        '`5617a9f61b028005a4858fdac845db406aefb181`. Официальное описание: '
        'https://huggingface.co/BAAI/bge-m3. Нормированный CLS-вектор, без префиксов E5. '
        'Пилот: `prepare_bge_m3_v10.py` → `encode_bge_m3_v10.py` → `bge_ranker_v10.py`. '
        'Загрузка скачивает только публичные веса; кодирование и обучение выполняются локально. '
        'Запросы и объявления не передаются внешнему сервису. Замороженные query-векторы вычислены '
        'из текста и фильтров, без query_id или test-разметки. Если канал выбран, его веса и векторы '
        'включены в архив; notebook использует сохранённые векторы для точного CPU-воспроизведения.\n'])
    joint=read(ROOT/'artifacts/cross-finetune-v10/joint_selection.json')
    if joint:
        lines.extend(['## Cross-encoder в итоговой комбинации\n',
            table(ROOT/'artifacts/cross-finetune-v10/joint_development.csv'),
            'Этот пилот сравнивался с вариантом до контрольного исключения дополнительного '
            'ранкера истории; пилот не выбран, потому что ухудшил development при всех '
            'ненулевых весах.\n',
            'Измеряется дополнение к уже выбранному v10, а не только к v9. Development-счёты '
            'вычислены локально CUDA BF16. При включении канала notebook действительно обрабатывает '
            'query и документы моделью на top-200 v9+category; готовые test-ответы или test-логиты '
            'не подставляются. CUDA BF16 повторяет режим пилота. CPU float32 поддерживается, '
            'но существенно медленнее; побайтовая проверка указывает фактически проверенное устройство.\n'])
    errors=read(CACHE/'error_analysis.json')
    if errors:
        lines.extend(['## Оставшиеся ошибки\n',
            'Пропуски известных положительных разделены на непопадание в широкий пул и ошибки выбора top-50. '
            'Источник меток — только train/development. Подробности: '
            '`artifacts/quality-v10/development_misses.csv`. Не выбранные пользователем объявления '
            'не объявляются доказанными отрицательными.\n','```json\n'+json.dumps(errors,indent=2)+'\n```\n'])
    lines.extend(['## Ограничения\n',
        'Development и контроль использовались в прежних экспериментах, они не являются новым '
        'независимым тестом. Сохранено известное ограничение старых вспомогательных priors v4. '
        'Удаление 90% положительных items из истории — допущение, а не известное устройство benchmark. '
        'Невыбранные объявления не считаются доказанно нерелевантными. Метрику скрытого теста '
        'может установить только платформа.\n',
        'Сторонний код или ответы кандидатов не копируются. Используются локальные open-source '
        'E5, mMARCO cross-encoder, Transformers, PyTorch, LightGBM и библиотеки из requirements.txt.\n'])
    (ROOT/'docs/EXPERIMENTS_V10.md').write_text('\n'.join(lines),encoding='utf-8')
    if manifest:
        final_metrics=manifest['selection']['winner']
        final_control=manifest['control']['selected_recall50']
        if manifest.get('field_selection'):
            final_metrics=manifest['field_selection']['winner']
            measured=read(ROOT/'artifacts/field-ranker-v10/control.json')
            if measured:final_control=measured['selected_recall50']
        if manifest.get('bge_selection'):
            final_metrics=manifest['bge_selection']['winner']
            measured=read(ROOT/'artifacts/bge-m3-v10/control.json')
            if measured:final_control=measured['selected_recall50']
        if manifest.get('warm_aux_selection'):
            final_metrics=manifest['warm_aux_selection']['winner']
            measured=read(CACHE/'warm_aux_control.json')
            if measured:final_control=measured['selected_recall50']
        if manifest.get('cross_selection'):
            final_metrics=manifest['cross_selection']['winner']
            measured=read(ROOT/'artifacts/cross-finetune-v10/joint_control.json')
            if measured:final_control=measured['selected_recall50']
        navigation=(
            '\n<!-- QUALITY_V10_START -->\n## Следующая отправка: v10\n\n'
            'Лучшая подтверждённая платформа: **v9 — 0.900753**. Новый кандидат ещё не отправлен.\n\n'
            f"- CSV: `{manifest['answer_file']}`.\n"
            '- Notebook кандидата: `experiments/Avito_v10_candidate.ipynb`.\n'
            '- Готовый комплект для проверяющего: `deliverables/avito_v10_solution.zip`.\n'
            '- Измерения, абляции и команды обучения: [EXPERIMENTS_V10.md](docs/EXPERIMENTS_V10.md).\n'
            f"- Локальный matched Recall@50: **{final_metrics['matched_recall50']:.6f}**; "
            f"контроль: **{final_control:.6f}**. Эти результаты не являются метрикой платформы.\n"
            f"- SHA-256 CSV: `{manifest['answer_sha256']}`.\n\n"
            'Корневые `Avito.ipynb` и `answer.csv` сохраняют v7; для следующей отправки используйте '
            'CSV v10 по указанному пути. В архиве v10 его имя уже `answer.csv`. '
            'Отправьте сначала один выбранный вариант; оставшиеся попытки используйте после получения '
            'обратной связи платформы.\n<!-- QUALITY_V10_END -->\n')
        path=ROOT/'README.md';existing=path.read_text(encoding='utf-8')
        begin=existing.find('\n<!-- QUALITY_V10_START -->')
        if begin>=0:
            end=existing.index('<!-- QUALITY_V10_END -->',begin)+len('<!-- QUALITY_V10_END -->')
            existing=existing[:begin]+existing[end:]
        title,_,body=existing.partition('\n')
        path.write_text(title+'\n'+navigation+'\n'+body.lstrip('\n'),encoding='utf-8')
        (ROOT/'docs/NEXT_SUBMISSION.md').write_text(navigation.replace('(docs/EXPERIMENTS_V10.md)','(EXPERIMENTS_V10.md)'),encoding='utf-8')
    notebook_path=ROOT/'experiments/Avito_neural_experiments.ipynb'
    nb=json.loads(notebook_path.read_text(encoding='utf-8'))
    nb['cells']=[c for c in nb['cells'] if c.get('id') not in {'quality-v10-note','quality-v10-control'}]
    nb['cells'].append({'cell_type':'markdown','metadata':{},'id':'quality-v10-note','source':[
        '## Эксперименты v10\n','\nВсе измерения: docs/EXPERIMENTS_V10.md. ',
        'Обучение и нейросетевые пилоты запускаются явно; по умолчанию выключены. ',
        'Первым выполняется категорийный эксперимент, затем расширенное обучение.\n']})
    code='''RUN_QUALITY_V10 = False
if RUN_QUALITY_V10:
    import subprocess, sys
    for script in ['quality_v10.py', 'train_quality_v10.py']:
        subprocess.run([sys.executable, str(ROOT / '.development' / script)], cwd=ROOT, check=True)
else:
    import json
    for name in ['category_selection.json', 'selection.json', 'control.json', 'reproduction.json']:
        path = ROOT / 'artifacts' / 'quality-v10' / name
        if path.exists():
            print(name, json.dumps(json.loads(path.read_text(encoding='utf-8')), indent=2))
'''
    nb['cells'].append({'cell_type':'code','metadata':{},'id':'quality-v10-control','source':code.splitlines(True),
        'execution_count':None,'outputs':[]})
    notebook_path.write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding='utf-8')
    print('Quality v10 report updated.')


if __name__=='__main__':report()
