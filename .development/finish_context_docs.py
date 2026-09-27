"""Update navigation only after the candidate passes offline reproduction."""
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parents[1]
manifest=json.loads((ROOT/'artifacts/context-v9/manifest.json').read_text())
proof=json.loads((ROOT/'artifacts/context-v9/reproduction.json').read_text())
assert proof['answer_bytes_equal'] and proof['network_connections_disabled']
assert proof['answer_sha256']==manifest['answer_sha256']
winner=manifest['context_selection']['winner']
control=manifest['context_control']['selected_recall50']
platform_path=ROOT/'artifacts/context-v9/platform_result.json'
platform=json.loads(platform_path.read_text()) if platform_path.exists() else None
platform_label=f"**{platform['recall50']:.6f}**" if platform else 'Не отправлена'
if platform:assert platform['answer_sha256']==manifest['answer_sha256']
readme=ROOT/'README.md'
text=readme.read_text(encoding='utf-8')
title='## Как работаем дальше' if '## Как работаем дальше' in text else '## Как работать с текущими версиями'
start=text.index(title)
end=text.index('Основной notebook и ответ',start)
replacement=f'''## Как работать с текущими версиями

| Версия | Где находится ответ | Платформа | Development / контроль |
| --- | --- | --- | --- |
| v7, подтверждённая | `answer.csv` | **0.897443** | 0.950171 / 0.953333 |
| v8, отдельный кандидат | `experiments/results/v8/answer.csv` | Не отправлена | 0.952815 / 0.953333 |
| v9, {'лучшая подтверждённая' if platform else 'новый кандидат'} | `{manifest['answer_file']}` | {platform_label} | {winner['matched_recall50']:.6f} / {control:.6f} |

Для проверки v9 на платформе используйте только `experiments/results/v9/answer.csv`.
Его notebook — `experiments/Avito_v9_candidate.ipynb`, комплект с локальными весами —
`deliverables/avito_v9_solution.zip`. Свежий запуск на CPU без сети получил
побайтово тот же CSV за {proof['elapsed_seconds']:.2f} секунд.
{'Платформа подтвердила рост до ' + platform_label + '; локальный контроль при этом ухудшился.' if platform else 'Контроль v9 ухудшился: локальный выигрыш небольшой и не гарантирует роста платформы.'}
Подтверждённые корневые notebook и CSV сохраняют v7.

Ветка v9 — `feature/context-v9`, подробный отчёт —
[docs/EXPERIMENTS_V9.md](docs/EXPERIMENTS_V9.md).
Предыдущие эксперименты — [docs/EXPERIMENTS_V8.md](docs/EXPERIMENTS_V8.md).

'''
readme.write_text(text[:start]+replacement+text[end:],encoding='utf-8')
changelog=ROOT/'CHANGELOG.md'
text=changelog.read_text(encoding='utf-8')
text=text[text.index('## v0.8.0-candidate'):]
entry=f'''## v0.9.0-candidate — контекстные признаки и более широкое обучение

- Ветка `feature/context-v9`; корневые v7 и отдельный v8 сохранены.
- Самостоятельные географические агрегаты с уверенностью и семантический перенос
  микрокатегорий/векторов от соседних запросов; `search_category` вне scoring.
- Четыре абляции на 4000 контекстов, CatBoost YetiRank, затем три варианта на
  12000 контекстов и 17 814 268 OOF-парах. В итог выбран `{winner['variant']}`
  с весом {winner['weight']}; остальные варианты не включены.
- Matched development: 0.952815 → {winner['matched_recall50']:.6f}; ранее просмотренный
  контроль: 0.953333 → {control:.6f}. {'Платформа: ' + platform_label + ' (результат сообщил пользователь).' if platform else 'На платформе v9 пока не проверен.'}
- CPU без сети и без готового CSV: {proof['elapsed_seconds']:.2f} секунд,
  {proof['code_cells_executed']} выполненных ячеек, побайтовое совпадение.
- Ответ `{manifest['answer_file']}`, notebook `experiments/Avito_v9_candidate.ipynb`,
  комплект `deliverables/avito_v9_solution.zip`; SHA-256 `{manifest['answer_sha256']}`.
- Данные, ошибки, протокол и ограничения — `docs/EXPERIMENTS_V9.md`.

'''
changelog.write_text(entry+text,encoding='utf-8')
print('Context v9 navigation updated after verified reproduction.')
