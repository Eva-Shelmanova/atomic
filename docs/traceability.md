# Соответствие требованиям задания

Матрица связывает 13 технических требований задания на производственную практику с компонентами, этапами дорожной карты, тестами и текущим статусом. Обновляется вместе с журналом.

Статусы: ✅ выполнено и покрыто тестами · 🟡 частично / в работе · ⬜ не начато.

| № | Требование | Компонент (код) | Этап | Тесты | Статус |
|---|---|---|---|---|---|
| 1 | Сохранить работоспособность базового ассистента | тег `v0.1-baseline`; регрессионные тесты `tests/test_atomic.py` | A, далее постоянно | `pytest -m "not slow"` (маркер `llm`) | ✅ зелёный на v0.1–v0.4 (режим shadow не меняет ответы чата; диагностика — отдельный метод `diagnose`) |
| 2 | Загружать и нормализовать временные ряды NPPAD | `atomic/diagnostics/nppad.py`, `scripts/load_nppad.py`, `config/channels.toml` | A | `tests/unit/test_nppad_loader.py` | ✅ |
| 3 | Единая семантическая модель диагностической информации | `atomic/knowledge/schema.py`, `builders.py`; `atomic/contracts/` | A | `tests/unit/test_knowledge.py`, `test_contracts.py` | ✅ |
| 4 | Гипотезы по ограниченному набору разрешённых каналов | `atomic/diagnostics/features.py`, `generator.py`, `scripts/train_generator.py`; `config/diagnostics.toml` [generator] | B | `tests/unit/test_generator.py` | ✅ test: top-1 0,888, top-3 0,996, ECE 0,028 |
| 5 | Генератор не видит скрытых данных и ground truth | `atomic/diagnostics/barrier.py`, `RestrictedWindow`, `labels.py` отдельно | A–B | `tests/contract/test_barrier.py`, `test_generator_rejects_unrestricted_window` | ✅ включая статическую проверку импортов генератора |
| 6 | Независимая проверка гипотез по полному набору каналов | `signatures.py`, `hypothesis_verifier.py` (окно + ранняя фаза, попарный отбор), `novelty.py`, `pipeline.py`; `scripts/train_verifier.py`, `train_novelty.py` | B–C | `test_hypothesis_verifier.py`, `test_onset.py`, `test_novelty.py` | ✅ test: верных подтверждено 0,969, неверных 0,015; точность IDENTIFIED 0,999 |
| 7 | Свидетельства «за»/«против», ссылки на аналоги событий | `ChannelEvidence` со ссылкой `<окно>#<канал>`; `analogues.py` (исходное событие + BM25), `scripts/load_ethz.py`, `config/analogues.toml`, `atomic/memory/bm25.py` | B–C | `test_hypothesis_verifier.py`, `test_analogues.py`, `test_bm25.py`, `test_diagnose_e2e.py` | ✅ precision@3 аналогов 0,936 |
| 8 | Передача гипотез в конвейер ассистента | `AnalystAgent.explain_diagnosis`, `atomic/agents/diagnosis_text.py`, `AtomicWorkflow.diagnose`, `POST /diagnose` | C | `test_diagnosis_answer.py`, `test_diagnose_e2e.py` | ✅ ссылки на базу знаний — механизм готов, документов по авариям в базе нет (D-040) |
| 9 | Детерминированная проверка итогового ответа | `atomic/verification/`, `config/rulepacks/aes-core-1.1.0.toml` (+ DIAGNOSIS_CONFIRMED, NO_DIRECTIVES), шлюз в `AtomicWorkflow` | B–C | `test_answer_verifier.py`, `test_workflow_verification.py`, `test_diagnosis_answer.py` | ✅ 1680 испорченных ответов — все BLOCK |
| 10 | Трассируемость и аудит | трасса `atomic.diagnose` в Langfuse: спаны генератора, проверки, новизны, документов + VerificationBundle; граф знаний (`builders.add_bundle`) | C | `test_diagnose_e2e.py` (с `ATOMIC_TEST_LANGFUSE=1` — с отправкой) | ✅ |
| 11 | Компонентное и сквозное тестирование | `tests/unit`, `tests/contract`, сквозной `test_diagnose_e2e.py` | B–C | все | ✅ см. `test-results/2026-10-25-stage-c.txt` |
| 12 | Benchmark генерации и проверки | `scripts/benchmark.py` (test, пробы, Variable_Power, верификатор ответа, аналоги, задержка), `eval_novelty_loco.py`, `eval_llm_narrative.py` | C | — | ✅ первый прогон на test (`test-results/2026-10-25-benchmark-test.json`); финальный — этап E |
| 13 | Воспроизводимость | `uv.lock`, `config/*.toml`, seed, паспорт версий в каждом результате, `benchmark.py --repeat` | C, проверка в E | `test_build_is_deterministic`, `--repeat` | ✅ повтор идентичен; проверка из чистого окружения — этап E |
