# Поиск кандидатов услуг Avito

**Итоговый Recall@50 на платформе: 0.914682.** Это версия v11; предыдущая
подтверждённая версия v10 получила 0.911829. Корневые
[`solution.ipynb`](solution.ipynb) и [`answer.csv`](answer.csv) относятся
именно к отправленной версии v11.

## С чего начать

- [Описание решения](docs/SOLUTION.md) — данные, признаки, модели, валидация,
  анализ ошибок и воспроизведение.
- [Notebook](solution.ipynb) — пошаговый код применения и сохранения ответа.
- [Эксперименты v11](docs/EXPERIMENTS_V11.md) — абляции и ограничения локальной
  оценки.
- [Скачать solution_v11.zip (Google Drive)](https://drive.google.com/file/d/1tTm7Tkl_Mw-T3A8cxxqlqpwbRLjXV4hN/view?usp=drive_link) —
  переносимый комплект для проверяющего с notebook, весами, кодом и отправленным
  CSV. Размер архива — 4.61 ГБ; SHA-256:
  `337da2857e4e89d3a2527020c7828e19256aad63e38b8252fbd26e31651fd508`.

Чтобы повторить ответ, распакуйте комплект, положите в его корень исходные
`train.parquet`, `benchmark_queries.parquet` и `benchmark_items.parquet`,
установите `requirements.txt` и выполните `solution.ipynb` целиком через
**Restart Kernel → Run All**. Используются только локальные файлы. Notebook
проверяет SHA-256 выходного `answer.csv`:
`8ce7401347beb1806f962ae1e72999fa93c89201b1044cd2433494f50e837d7e`.

В `experiments/` и `archive/notebooks/` сохранены предыдущие версии для
аудита; все их notebook-файлы имеют префикс `solution`. Код новых признаков
и обучения находится в `.development/`. В Git нет исходных Parquet-файлов и
больших весов; для полного воспроизведения нужен переносимый комплект.
