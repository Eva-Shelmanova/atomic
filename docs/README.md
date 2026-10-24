# Документация дипломной работы atomic

Тема практики: «Интерактивный ассистент для поддержки принятия решений операторами АЭС на основе формирования и верификации гипотез о состоянии системы».

План работ: `books/Бакалаврский диплом техническая дорожная карта.pdf` (этапы A–F, сентябрь 2026 — май 2027).

## Документы

| Файл | Что внутри | Куда пойдёт |
|---|---|---|
| [full-report.md](full-report.md) | Полный отчёт простым языком: от 6-го семестра до заморозки — действия, решения, результаты, ограничения | основа введения и глав ВКР, материал для защиты |
| [journal.md](journal.md) | Журнал работ по датам: что сделано, зачем, результат, коммит | хронология для отчётов, отзыв руководителя |
| [decisions.md](decisions.md) | Реестр проектных решений D-001… с альтернативами и обоснованием | разделы «Обоснование выбора», ответы оппонентам |
| [traceability.md](traceability.md) | Матрица: 13 требований задания → модули → тесты → статус | раздел «Соответствие заданию» |
| [datasets.md](datasets.md) | Источники данных (NPPAD, ETHZ): версии, лицензии, состав, статистика | раздел «Исходные данные» |
| [stage-A-data-model.md](stage-A-data-model.md) | Модель данных и методы (этап A, отчётность 31.10.2026) | глава «Модель данных» отчёта практики и ВКР |
| [stage-B-methods.md](stage-B-methods.md) | Методы формирования и проверки гипотез, промежуточное тестирование (этап B) | глава «Программная реализация» |
| [stage-C-integration.md](stage-C-integration.md) | Интеграция с ассистентом, новизна, benchmark на test, количественные показатели (этап C) | главы «Программная реализация» и «Эксперименты» |
| [stage-E-freeze.md](stage-E-freeze.md) | Доводка и заморозка: исправленные дефекты, защёлка события, справка по сценариям, LLM-as-a-Judge, воспроизводимость из чистого окружения, финальные числа | главы «Эксперименты» и «Заключение» ВКР |
| [data-audit.md](data-audit.md) | Аудит классов NPPAD (пустые и однопрогонные классы) и полей ETHZ | раздел «Исходные данные», ограничения |
| [test-results/](test-results/) | Сохранённые прогоны тестов с датой и коммитом | приложения отчёта |

## Документы для сдачи

Папка [`../reports/`](../reports/README.md): контрольная точка этапа A, отчёт по практике по ГОСТ 7.32 и полный отчёт в формате .docx, их исходники и генератор.

## Как воспроизвести

```bash
uv sync
git clone --depth 1 https://github.com/thu-inet/NuclearPowerPlantAccidentData.git NPPAD/raw
uv run python scripts/load_nppad.py              # подготовка NPPAD одной командой
uv run python scripts/load_ethz.py               # база событий ETHZ (нужна папка ETHDB/)
uv run python scripts/train_generator.py         # генератор гипотез (≈ 40 с)
uv run python scripts/train_verifier.py          # сигнатуры верификатора гипотез (≈ 1 с)
uv run python scripts/train_novelty.py           # детектор новизны (≈ 15 с)
uv run python scripts/build_scenario_kb.py       # справка по сценариям из журналов PCTRAN (dataset/scenarios)
uv run python scripts/convert_variable_power.py  # стресс-тест изменения мощности (.mdb → CSV)
uv run python scripts/evaluate_stage_b.py        # промежуточная оценка на val (≈ 5 мин)
uv run python scripts/benchmark.py --repeat      # итоговый benchmark на test + проверка повтора (≈ 5 мин)
uv run python scripts/eval_novelty_loco.py       # предел проверки новизны («исключить один класс», val)
uv run python scripts/eval_llm_narrative.py --judge   # пояснения LLM + LLM-as-a-Judge (нужна Ollama)
uv run python scripts/eval_analogues.py          # precision@k / recall@k аналогов ETHZ
uv run python scripts/export_diagnose_trace.py   # живая трасса в Langfuse (docker compose -f langfuse/docker-compose.yml up -d)
uv run python scripts/compare_benchmarks.py A.json B.json   # совпадение двух прогонов benchmark
uv run python scripts/demo_diagnosis.py --scenario LOCA --t 300   # пример для отчёта
uv run pytest -m "not slow and not llm"          # модульные и контрактные тесты без LLM
uv run pytest -m "not slow"                      # + регрессия базового ассистента (нужна Ollama)
```

Версия Python закреплена в `.python-version` (3.12), зависимости — в `uv.lock`. Если запущен Docker Desktop с контейнером `ollama`, он занимает порт 11434 и `localhost` указывает на него; для системного Ollama задайте `OLLAMA_HOST=http://127.0.0.1:11434`.

## Версии (git-теги)

| Тег | Что зафиксировано |
|---|---|
| `v0.1-baseline` | Ассистент 6-го семестра (коммит a3f0e16) — точка отсчёта требования 1 |
| `v0.2-stage-a` | Завершение этапа A: данные, семантическая модель, контракты, барьер |
| `v0.3-stage-b` | Завершение этапа B: генератор и верификатор гипотез, аналоги ETHZ, верификатор ответа |
| `v0.4-stage-c` | Завершение этапа C: интеграция с AnalystAgent, память о событии, новизна, Langfuse, benchmark на test |
| `v1.0-freeze` | Заморозка (этап E): финальные числа для ВКР; дальше код меняется только для исправления ошибок |
