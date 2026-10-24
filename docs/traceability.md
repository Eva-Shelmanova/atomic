# Соответствие требованиям задания

Матрица связывает 13 технических требований задания на производственную практику с компонентами, этапами дорожной карты, тестами и текущим статусом. Обновляется вместе с журналом.

Статусы: ✅ выполнено и покрыто тестами · 🟡 частично / в работе · ⬜ не начато.

| № | Требование | Компонент (код) | Этап | Тесты | Статус |
|---|---|---|---|---|---|
| 1 | Сохранить работоспособность базового ассистента | тег `v0.1-baseline`; регрессионные тесты `tests/test_atomic.py` | A, далее постоянно | `pytest -m "not slow"` (маркер `llm`) | ✅ зелёный на v0.1–v0.4 (режим shadow не меняет ответы чата; диагностика — отдельный метод `diagnose`) |
| 2 | Загружать и нормализовать временные ряды NPPAD | `atomic/diagnostics/nppad.py`, `scripts/load_nppad.py`, `config/channels.toml` | A | `tests/unit/test_nppad_loader.py` | ✅ |
| 3 | Единая семантическая модель диагностической информации | `atomic/knowledge/schema.py`, `builders.py`; `atomic/contracts/` | A | `tests/unit/test_knowledge.py`, `test_contracts.py` | ✅ |
| 4 | Гипотезы по ограниченному набору разрешённых каналов | `atomic/diagnostics/features.py`, `generator.py`, `scripts/train_generator.py`; `config/diagnostics.toml` [generator] | B | `tests/unit/test_generator.py` | ✅ финал (test): top-1 0,888, top-3 0,996, ECE 0,028 |
| 5 | Генератор не видит скрытых данных и ground truth | `atomic/diagnostics/barrier.py`, `RestrictedWindow`, `labels.py` отдельно | A–B | `tests/contract/test_barrier.py`, `test_generator_rejects_unrestricted_window` | ✅ включая статическую проверку импортов генератора |
| 6 | Независимая проверка гипотез по полному набору каналов | `signatures.py`, `hypothesis_verifier.py` (окно + ранняя фаза, попарный отбор), `novelty.py`, `onset.py`, `pipeline.py`; `scripts/train_verifier.py`, `train_novelty.py` | B–E | `test_hypothesis_verifier.py`, `test_onset.py`, `test_novelty.py` | ✅ финал: верных подтверждено 0,969, неверных 0,015; точность IDENTIFIED 0,999 |
| 7 | Свидетельства «за»/«против», ссылки на аналоги событий | `ChannelEvidence` со ссылкой `<окно>#<канал>`; `analogues.py` (исходное событие + BM25), `scripts/load_ethz.py`, `config/analogues.toml`, `atomic/memory/bm25.py` | B–E | `test_hypothesis_verifier.py`, `test_analogues.py`, `test_bm25.py`, `test_diagnose_e2e.py` | ✅ текст: P@3 0,68, R@20 0,35; совмещённо P@3 0,93 |
| 8 | Передача гипотез в конвейер ассистента | `AnalystAgent.explain_diagnosis`, `diagnosis_text.py`, `AtomicWorkflow.diagnose`, `POST /diagnose`; справка по сценариям `dataset/scenarios` | C–E | `test_diagnosis_answer.py`, `test_diagnose_e2e.py`, `test_api_diagnose.py` | ✅ ссылки на телеметрию, ETHZ и справку по сценарию (D-043) |
| 9 | Детерминированная проверка итогового ответа | `atomic/verification/`, `config/rulepacks/aes-core-1.1.0.toml` (+ DIAGNOSIS_CONFIRMED, NO_DIRECTIVES), шлюз в `AtomicWorkflow` | B–C | `test_answer_verifier.py`, `test_workflow_verification.py`, `test_diagnosis_answer.py` | ✅ 1680 испорченных ответов — все BLOCK |
| 10 | Трассируемость и аудит | трасса `atomic.diagnose` в Langfuse: спаны этапов + VerificationBundle; `scripts/export_diagnose_trace.py` | C–E | `test_diagnose_e2e.py`; живая трасса `test-results/2026-10-25-langfuse-trace.json` | ✅ путь «данные → … → решение» восстанавливается по трассе |
| 11 | Компонентное и сквозное тестирование | `tests/unit`, `tests/contract`, сквозные `test_diagnose_e2e.py`, `test_api_diagnose.py`; LLM-as-a-Judge — `eval_llm_narrative.py --judge` | B–E | все | ✅ см. `test-results/2026-10-25-stage-e.txt` |
| 12 | Benchmark генерации и проверки | `scripts/benchmark.py`, `eval_novelty_loco.py`, `eval_llm_narrative.py`, `eval_analogues.py` | C–E | — | ✅ финал на замороженной версии `v1.0-freeze` (`test-results/2026-10-25-benchmark-final.json`) |
| 13 | Воспроизводимость | `uv.lock`, `.python-version`, `config/*.toml`, seed, паспорт версий (вкл. digest моделей Ollama), `benchmark.py --repeat`, `compare_benchmarks.py` | C–E | `--repeat`; чистое окружение | ✅ повтор и сборка с нуля из чистого клона совпадают (`test-results/2026-10-25-reproducibility-clean.json`) |
