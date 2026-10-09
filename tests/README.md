# Проверенные тесты

`python -m pytest -q` запускает 19 тестов реальных авторских классов.
Все проверки находятся в `test_transformer.py`: masks, causality, shapes, gradients,
dropout, positional buffers, tiny overfit и SHA-256 сохранённых исходников.
Полный план и ограничения проверки описаны в README и docs/review.md.
