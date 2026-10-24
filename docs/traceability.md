# Соответствие требованиям задания

Матрица связывает 13 технических требований задания на производственную практику с компонентами, этапами дорожной карты, тестами и текущим статусом. Обновляется вместе с журналом.

Статусы: ✅ выполнено и покрыто тестами · 🟡 частично / в работе · ⬜ не начато.

| № | Требование | Компонент (код) | Этап | Тесты | Статус |
|---|---|---|---|---|---|
| 1 | Сохранить работоспособность базового ассистента | тег `v0.1-baseline`; регрессионные тесты `tests/test_atomic.py` | A, далее постоянно | `pytest -m "not slow"` (маркер `llm`) | 🟡 базовый прогон 24.10 зелёный; повторяется после каждого этапа |
| 2 | Загружать и нормализовать временные ряды NPPAD | `atomic/diagnostics/nppad.py`, `scripts/load_nppad.py`, `config/channels.toml` | A | `tests/unit/test_nppad_loader.py` | ✅ |
| 3 | Единая семантическая модель диагностической информации | `atomic/knowledge/schema.py`, `builders.py`; `atomic/contracts/` | A | `tests/unit/test_knowledge.py`, `test_contracts.py` | ✅ |
| 4 | Гипотезы по ограниченному набору разрешённых каналов | `atomic/diagnostics/generator.py`; `config/diagnostics.toml` [generator] | B | — | ⬜ |
| 5 | Генератор не видит скрытых данных и ground truth | `atomic/diagnostics/barrier.py`, `RestrictedWindow`, `labels.py` отдельно | A–B | `tests/contract/test_barrier.py` | ✅ барьер; повторная проверка после генератора |
| 6 | Независимая проверка гипотез по полному набору каналов | `atomic/diagnostics/hypothesis_verifier.py` | B | — | ⬜ |
| 7 | Свидетельства «за»/«против», ссылки на аналоги событий | верификатор гипотез + `analogues.py` (ETHZ) | B | — | ⬜ |
| 8 | Передача гипотез в конвейер ассистента | интеграция с AnalystAgent | C | — | ⬜ |
| 9 | Детерминированная проверка итогового ответа | `atomic/verification/` (Claim → EvidenceRef → RulePack → PolicyDecision) | B | — | ⬜ |
| 10 | Трассируемость и аудит | VerificationBundle в Langfuse; граф знаний (`builders.add_bundle`) | C | — | ⬜ (подготовлены типы и граф) |
| 11 | Компонентное и сквозное тестирование | `tests/unit`, `tests/contract`, сквозные — этап C | B–C | все | 🟡 |
| 12 | Benchmark генерации и проверки | офлайн-оценщик на NPPAD | C | — | ⬜ |
| 13 | Воспроизводимость | `uv.lock`, `config/*.toml`, seed, манифест с коммитом источника и хэшем конфигурации | C, проверка в E | `test_build_is_deterministic` | 🟡 |
