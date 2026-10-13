# SymFw — Probe Design (two-pass delta)

Зафиксированное решение. Mem0-бэкенд в этой среде не персистит записи, поэтому
решение хранится здесь.

## Baseline и дельта
- `baseline = pool(hidden(system_prompt))` — текущий системный промпт, чистый (контролируем мы), кэш по `hash(system_prompt)`.
- `DELTA = pool(hidden(system + user)) - pool(hidden(system))`.
- Дельта всегда меряется **против текущего системного промпта**, не против замороженного шаблона → дельта есть всегда и она = «что user-ввод сделал с моделью относительно текущей цели».
- Оговорки:
  1. baseline должен быть чистым — иначе точка отсчёта отравлена;
  2. дельта ≠ вердикт — «обработка данных» vs «перехват» различает именно проб;
  3. кэш выгоден только при стабильном системном промпте, иначе 2 forward pass.

## Что такое проб (технически)
Маленькая **обученная** модель-классификатор поверх `DELTA`. НЕ key-value, НЕ threshold. LLM заморожена, проб «едет» на её hidden states.

- **Линейный проб** (`probe_detector.py` = sklearn `LogisticRegression`):
  `p(inj) = sigmoid(w · DELTA + b)`.
  `w` — выученная **ось перехвата**; `cos(DELTA, w)` = сигнал. Гиперплоскость, `w` интерпретируем.
- **Threshold** — отдельная ручка поверх вероятности (ROC-настройка под цену ошибки; для security — высокий recall). НЕ часть модели.
- **Альтернативы**:
  - MLP 2 слоя — нелинейная граница;
  - LDA — дискриминантная ось;
  - Mahalanobis / kNN к чистым дельтам — «key-value»-подобный unsupervised anomaly (не нужны размеченные инъекции; минус — tricky-легитимные тоже далеко → FP).
- **Обучение**: supervised, пары `(DELTA, label)`, сотни–несколько тысяч, `class_weight="balanced"`. Train секунды на CPU, инференс <1ms (один matmul).
- **MVP-выбор**: линейный проб (интерпретируемый `w` показывает ось перехвата). Mahalanobis — бонус, если нет размеченных инъекций.

## Task-зависимость дельт
- Сырые дельты разные под каждую задачу/контент (разный baseline = другая система координат).
- Но **направление перехвата почти shared** (свойство модели).
- Task-устойчивость проба: diverse mix задач при обучении; нормализация `d / (||e_base|| + eps)`; угол к hijack-прототипу (инвариантно к повороту baseline); whitening по within-class ковариации; fine-tune 50–100 примеров на новую задачу.
- Смена модели = переобучать обязательно (разные hidden-пространства).

## Связь с кодом
- `two_pass_detector.py` — baseline-кэш, дельта, `hijack_projection`, `mock_provider`.
- `probe_detector.py` — `LogisticRegression` + `build_delta_dataset(provider, system, samples)`.
- `decision_engine.py` — вердикт = `probe_prob` + маркеры; сырые `drift`/`delta_norm` вес 0 (не дискриминантны).
- `firewall.py` — оркестратор `check(system_prompt, user_input)`.

## Конвейеры обучения (два)

### Mock-runnable (`probe_trainer.py`)
Валидирует механику без GPU/зависимостей (numpy-only: scaler/PCA/logreg свои).
Прогоняется end-to-end: `build_dataset` (48 промптов) → stratified split →
scaler → PCA → logreg (class-balanced GD) → калибровка порога по recall →
метрики (AUROC, FPR@recall, per-group) → `pickle` → загрузка в firewall → `check()`.
Запуск: `cd /root && python -m SymFw.probe_trainer`.
Результат на mock: test AUROC 1.0, thr 0.936, recall 1.0, FPR 0.0;
tricky mean_p=0.261 > clean 0.068, но под порогом → 0 FP (сепарация по
направлению, не по величине). Артефакт: `probe_mock.pkl`.

### Real / GPU scaffold (`probe_trainer_hf.py`)
Боевая заготовка под реальную LLM. НЕ запускается без torch/transformers/sklearn.
- `HFProvider`: `AutoModel(output_hidden_states=True)`, masked pooling
  (`last`/`mean`/`weighted`) по выбранным слоям, конкатенация → `[len(layers)*d]`.
- `build_deltas`: two-pass, baseline кэш по system_prompt, нормализация `d/||e_base||`.
- sklearn `Pipeline`: `StandardScaler → PCA(whiten) → LogisticRegression(class_weight=balanced)`.
- Калибровка: `roc_curve`, порог = макс. при `tpr >= target_recall`.
- Метрики: `roc_auc_score`, FPR@recall, per-group (clean/tricky/injection).
- `cross_task_eval`: train на task A, test на task B — переносимость.
- Артефакт: `pickle` {pipeline, threshold, meta(model_id, layers, pool, pca_dim, ...)}.
- CLI: `python -m SymFw.probe_trainer_hf --model Qwen/Qwen3-8B --layers 3 6 12
  --pool last --data labeled.jsonl --out probe_qwen3-8b.pkl --pca-dim 128 --target-recall 0.95`.
- Формат данных: JSONL `{"system","user","label","group"}`; инъекции — AdvBench/HarmBench/Garak.

### vLLM caveat
vLLM не отдаёт per-layer hidden states для произвольных промптов из коробки.
Обучение — через HF. Для продеплоя с vLLM: кастомный worker или отдельный
HF hidden-state экстрактор рядом с сервером. `VLLMProvider` — заглушка.

### Перенос mock → real
Конвейер идентичен; меняется только provider (`mock_provider` → `HFProvider`) и
проб (numpy → sklearn). Механика, сплиты, калибровка, артефакт — те же.
