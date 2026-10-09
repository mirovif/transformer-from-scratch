# Ревью предоставленной реализации

Сохранены авторские классы, Q/K/V и W_o проекции, ручное разделение heads,
матричное умножение attention, encoder/decoder blocks, ReLU FFN, post-norm residual
и синусоидальный positional encoding. `nn.Transformer`, `nn.MultiheadAttention`
и готовый scaled_dot_product_attention не используются.
Девять исходных файлов сохранены побайтно в `original/`; их SHA-256 — `original/manifest.json`.
Файлы учебных заданий из архива не включены.

| Наблюдение | Изменение | Проверка |
|---|---|---|
| `mask.unsqueeze_` изменял маску вызывающего кода; следующие блоки получали другой rank | не изменяющая вход нормализация 2D/3D/4D masks | повторное использование одной маски, shape до/после |
| без `tgt_mask` decoder видел будущие токены | causal mask автоматически, объединение с target padding | изменение будущего токена не меняет ранние logits |
| source PAD-токены не закрывались по умолчанию | автоматический key padding mask в encoder и cross attention | изменение masked key/value не меняет attention output |
| полностью закрытая строка давала softmax от `-inf` и NaN | finite minimum + нулевые masked weights | finite forward/backward при полном mask |
| `heads=0` вызывал деление на ноль | явный ValueError | некорректное число heads отклоняется |
| слишком длинный вход падал при сложении с PE | явная проверка `0 < T <= max_len` | проверка длины |
| не было обучения и генерации | отдельные copy-task training и greedy decoding | реальный запуск с отдельными train/validation/test |

Имена файлов приведены к snake_case, imports стали относительными.
Обучение вызывает авторский `Transformer.forward`, без заменяющей реализации.

## Ограничения и предложения

- Dropout остаётся в embedding, как в авторском коде. Attention/residual dropout и
  label smoothing из оригинальной статьи не добавлены автоматически.
- LayerNorm реализован после residual (post-norm); это зафиксировано в документации.
- Полностью masked attention имеет нулевую взвешенную сумму V, но W_o bias может
  давать ненулевой выход. PAD-позиции исключены из loss.
- Проверена малая CPU-конфигурация и задача копирования, а не machine translation.
- Нет tokenizer, реального корпуса, beam search, KV-cache или mixed precision.
- Добавление dropout в подслои, scheduler/warmup и real-text tokenizer — отдельные
  будущие эксперименты, чтобы не смешивать исправление ошибок со сменой архитектуры.
- Нельзя утверждать полное соответствие статье: отличаются training recipe и dropout.
