# Соответствие требованиям задания

Матрица связывает 13 технических требований задания на производственную практику с компонентами, этапами дорожной карты, тестами и текущим статусом. Обновляется вместе с журналом.

Статусы: ✅ выполнено и покрыто тестами · 🟡 частично / в работе · ⬜ не начато.

| № | Требование | Компонент (код) | Этап | Тесты | Статус |
|---|---|---|---|---|---|
| 1 | Сохранить работоспособность базового ассистента | тег `v0.1-baseline`; регрессионные тесты `tests/test_atomic.py` | A, далее постоянно | `pytest -m "not slow"` (маркер `llm`) | 🟡 зелёный на v0.1, v0.2 и после B3 (режим shadow не меняет ответы) |
| 2 | Загружать и нормализовать временные ряды NPPAD | `atomic/diagnostics/nppad.py`, `scripts/load_nppad.py`, `config/channels.toml` | A | `tests/unit/test_nppad_loader.py` | ✅ |
| 3 | Единая семантическая модель диагностической информации | `atomic/knowledge/schema.py`, `builders.py`; `atomic/contracts/` | A | `tests/unit/test_knowledge.py`, `test_contracts.py` | ✅ |
| 4 | Гипотезы по ограниченному набору разрешённых каналов | `atomic/diagnostics/features.py`, `generator.py`, `scripts/train_generator.py`; `config/diagnostics.toml` [generator] | B | `tests/unit/test_generator.py` | ✅ val: top-1 0,786, top-3 0,998, ECE 0,029 |
| 5 | Генератор не видит скрытых данных и ground truth | `atomic/diagnostics/barrier.py`, `RestrictedWindow`, `labels.py` отдельно | A–B | `tests/contract/test_barrier.py`, `test_generator_rejects_unrestricted_window` | ✅ включая статическую проверку импортов генератора |
| 6 | Независимая проверка гипотез по полному набору каналов | `atomic/diagnostics/signatures.py`, `hypothesis_verifier.py`, `scripts/train_verifier.py` (82 наблюдаемых канала) | B | `tests/unit/test_hypothesis_verifier.py` | ✅ |
| 7 | Свидетельства «за»/«против», ссылки на аналоги событий | `ChannelEvidence` со ссылкой `<окно>#<канал>`; `analogues.py`, `scripts/load_ethz.py`, `config/analogues.toml`, `atomic/memory/bm25.py` | B | `test_hypothesis_verifier.py`, `test_analogues.py`, `test_bm25.py` | ✅ |
| 8 | Передача гипотез в конвейер ассистента | интеграция с AnalystAgent | C | — | ⬜ |
| 9 | Детерминированная проверка итогового ответа | `atomic/verification/`, `config/rulepacks/aes-core-1.0.0.toml`, шлюз в `AtomicWorkflow` | B | `tests/unit/test_answer_verifier.py`, `test_workflow_verification.py` | ✅ |
| 10 | Трассируемость и аудит | VerificationBundle в метаданных трассы Langfuse (`ask_verified`); граф знаний (`builders.add_bundle`) | C | — | 🟡 bundle пишется в трассу ответа; трасса диагностического контура — этап C |
| 11 | Компонентное и сквозное тестирование | `tests/unit`, `tests/contract`, сквозные — этап C | B–C | все | 🟡 |
| 12 | Benchmark генерации и проверки | офлайн-оценщик на NPPAD; на этапе B — промежуточная оценка на val (`scripts/evaluate_stage_b.py`, `sweep_verifier.py`) | C | — | 🟡 метрики на val; test зарезервирован (D-025) |
| 13 | Воспроизводимость | `uv.lock`, `config/*.toml`, seed, манифест с коммитом источника и хэшем конфигурации | C, проверка в E | `test_build_is_deterministic` | 🟡 |
