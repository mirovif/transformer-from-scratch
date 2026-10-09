# Архитектура проверенной авторской реализации

Классы из пользовательского архива проверены тестами и CPU-обучением на copy task.

`B` — batch size; `T` — sequence length; `D` — d_model; `V` — vocabulary size;
`h` — number of heads; `d_k = D / h`; `N` — число блоков.
`T_src` и `T_tgt` уточняют длину source и target, которые могут различаться.

| Узел | Вход → выход |
|---|---|
| Source/target tokens | `(B, T_src)` / `(B, T_tgt)` целочисленные IDs |
| Embedding + positional encoding | `(B, T)` → `(B, T, D)` |
| Q/K/V projections | `(B, T, D)` → `(B, h, T, d_k)` |
| Self-attention scores | `(B, h, T, d_k)` @ `(B, h, d_k, T)` → `(B, h, T, T)` |
| Cross-attention scores | `(B, h, T_tgt, T_src)` |
| Concatenated attention output | `(B, h, T, d_k)` → `(B, T, D)` |
| Feed Forward Network | `(B, T, D)` → `(B, T, D_ff)` → `(B, T, D)` |
| Vocabulary projection | `(B, T_tgt, D)` → logits `(B, T_tgt, V)` |

$$Attention(Q,K,V) = softmax(QK^T / \sqrt{d_k} + M)V$$

Multi-Head Attention выполняет attention в `h` проекциях и объединяет их выходы.
Encoder содержит self-attention и FFN; decoder добавляет masked self-attention
и cross-attention к памяти encoder. Residual connections складывают вход и выход
подслоя одинаковой формы. LayerNorm нормализует последнее измерение `D`.
В коде используется post-norm, ReLU, embedding scale sqrt(D), embedding dropout.
Attention/residual dropout отсутствуют.

Causal mask запрещает доступ к будущим target-токенам. Для additive mask
нижний треугольник равен 0, верхний равен `-inf`; mask применяется **до softmax**.
Padding mask отдельно закрывает PAD-ключи. В этой реализации boolean mask использует True для разрешённых позиций.
Полностью закрытые строки обрабатываются без NaN: masked weights принудительно обнуляются.
Поведение полностью закрытых строк проверено тестом на конечные forward и backward.

На обучении target обычно сдвинут: decoder получает BOS + предыдущие токены,
loss сравнивает logits с последующими токенами и исключает PAD.
`CrossEntropyLoss` принимает logits; softmax перед loss не нужен.
На инференсе токены генерируются последовательно до EOS или ограничения длины.

Оригинальная статья: [Attention Is All You Need](https://arxiv.org/abs/1706.03762).
Полное соответствие статье не заявляется; отличия перечислены в review.md.
