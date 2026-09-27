"""Write an honest summary from completed local experiment reports only."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]

def read(relative):
    path = ROOT/relative
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else None

def write_report():
    selection = read('artifacts/ranker-v8/selection.json')
    control = read('artifacts/ranker-v8/control.json')
    expansion = read('artifacts/expanded-pool-v8/pilot_report.json')
    pool = read('artifacts/learned-pool-v8/report.json')
    ce = read('artifacts/ce-feature-v8/pilot_report.json')
    hard = read('artifacts/query-hard-negative-v8/pilot_report_7287a66ad89510fd.json')
    continuation = read('artifacts/query-hard-negative-v8/pilot_report_5bae8f3260cf556d.json')
    combined = read('artifacts/combined-pool-v8/selection.json')
    combined_control = read('artifacts/combined-pool-v8/control.json')
    candidate = read('artifacts/combined-pool-v8/manifest.json')
    reproduction = read('artifacts/combined-pool-v8/reproduction.json')
    report = {'confirmed_platform_recall50':.897443,'ranker_selection':selection,'ranker_control':control,
        'learned_pool':pool,'expanded_pool_final_recall':expansion,'hard_negative_pilot':hard,
        'continuation_pilot':continuation,'ce_feature_pilot':ce,
        'combined_selection':combined,'combined_control':combined_control,
        'candidate':candidate,'candidate_reproduction':reproduction,
        'primary_answer_sha256':hashlib.sha256((ROOT/'answer.csv').read_bytes()).hexdigest()}
    assert report['primary_answer_sha256']=='a256639b72ef3bd515bf5118c5e337aee23a34c9e0b261da580b1c852411a6b2'
    (ROOT/'artifacts/v8_progress.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    lines = ['# Эксперименты после подтверждённой v7\n',
        'Основной answer.csv: платформа **0.897443**, сохранён без изменения. '
        'Эксперименты доступны в experiments/Avito_neural_experiments.ipynb.\n',
        '| Эксперимент | Локальный результат |\n| --- | --- |']
    if selection:
        w = selection['winner'];b=selection['baseline']
        lines.append(f'| Усиление ранкера: {w["model"]}, {int(w["trees"])} деревьев, вес {w["weight"]} | '
            f'Development matched: {b["matched_recall50"]:.6f} → {w["matched_recall50"]:.6f} |')
    if control:
        lines.append(f'| Контроль выбранного ранкера | {control["v7_recall50"]:.6f} → {control["selected_recall50"]:.6f}; '
            f'улучшений {control["improved"]}, ухудшений {control["worsened"]} |')
    if hard:
        lines.append(f'| Дополнительная эпоха E5 с трудными отрицательными | Лучший development: {hard["winner"]["matched_recall50"]:.6f} |')
    if continuation:
        lines.append(f'| Такая же эпоха без нового loss | Лучший development: {continuation["winner"]["matched_recall50"]:.6f}; '
            'отдельная польза трудных отрицательных примеров не подтверждена |')
    if pool:
        p = [r for r in pool['results'] if r['mode']=='unseen_text' and r['added_per_channel']==500][0]
        lines.append(f'| Поиск learned E5 по всему корпусу | Полнота пула {p["pool_recall"]:.6f}; '
            f'возвращено {p["rescued_positive_pairs"]}/{p["outside_positive_pairs"]} пропущенных пар |')
    if expansion:
        lines.append(f'| Финальные 50 из расширенного пула | Development matched: '
            f'{expansion["before_matched_recall50"]:.6f} → {expansion["after_matched_recall50"]:.6f} |')
    if ce:
        lines.append(f'| CE как признак обучаемого отбора | {ce["training_contexts"]} обучающих / '
            f'{ce["validation_contexts"]} проверочных контекстов; прирост {ce["ce_delta_vs_best_without"]:.6f} |')
    if combined:
        lines.append(f'| Лучшее сочетание расширенного пула и ранкеров | Development matched: '
            f'{combined["baseline_v7_matched"]:.6f} → {combined["winner"]["matched_recall50"]:.6f}; '
            f'вес усиленного ранкера {combined["winner"]["strong_ranker_weight"]} |')
    if combined_control:
        lines.append(f'| Контроль итогового сочетания | {combined_control["v7_recall50"]:.6f} → '
            f'{combined_control["combined_recall50"]:.6f} |')
    if combined and combined['winner']['strong_ranker_weight']==0:
        lines.append('\nВ итоговом кандидате сохранён ранкер v7 и расширена кандидатогенерация. '
            'Новый ранкер отдельно улучшил development и контроль, но на расширенном пуле '
            'его добавление проиграло прежнему ранкеру по основной matched-метрике. '
            'Он сохранён как резерв: `experiments/results/ranker_v8/answer.csv`.\n')
    lines += ['\n## Как интерпретировать результаты\n',
        'Development и контроль ранее просматривались. Старые вспомогательные priors '
        'имеют ограничение по использованной истории; evaluation query-encoder '
        'исключает все development-тексты. Это локальные сравнения, а не оценки '
        'новой метрики скрытого теста.\n',
        'Трудные отрицательные примеры исключают известные положительные, однако '
        'остальные объявления не являются достоверно отрицательными. CE проверен '
        'на небольшом разбиении по текстам; полный обученный cross-encoder не построен.\n',
        'Расширение пула оценивается отдельно от Recall@50. Применение прежнего '
        'ранкера к новому пулу не заменяет обучение на расширенных OOF-группах.\n']
    if candidate:
        lines += ['## Отдельный кандидат\n',
            f'CSV: `{candidate["answer_file"]}`. Его SHA-256: `{candidate["answer_sha256"]}`. '
            'Платформенная метрика пока неизвестна.\n']
    if reproduction:
        lines.append(f'Кандидат воспроизведён: CPU, сеть отключена, идентичный CSV; '
            f'{reproduction["elapsed_seconds"]:.2f} секунд.\n')
    lines += ['## Готовые файлы и оставшаяся работа\n',
        'Для следующей проверки подготовлен `experiments/results/v8/answer.csv`. '
        'Его notebook: `experiments/Avito_v8_candidate.ipynb`. Переносимый комплект '
        'с одним основным notebook, моделями и CSV: `deliverables/avito_v8_solution.zip`. '
        'Корневые Avito.ipynb и answer.csv остаются подтверждённой версией v7.\n',
        'Это завершённый блок ограниченных экспериментов, а не закрытие всех направлений плана. '
        'Ранкер использует прежние OOF-пулы и признаки; новые отрицательные примеры, выбранные '
        'сильным OOF-ранкером, и обучение на расширенных OOF-пулах здесь не реализованы. '
        'Новый большой encoder, полное дообучение cross-encoder, дополнительные географические '
        'признаки и новые политики холодных объявлений требуют отдельных экспериментов.\n']
    (ROOT/'docs/EXPERIMENTS_V8.md').write_text('\n'.join(lines).rstrip()+'\n',encoding='utf-8')
    print('Updated docs/EXPERIMENTS_V8.md and artifacts/v8_progress.json.')

if __name__=='__main__':
    write_report()
