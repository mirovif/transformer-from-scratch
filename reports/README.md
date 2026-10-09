# Результаты обучения

`summary.json` — конфигурация, параметры, validation и отдельный final test.
`training_history.csv` — training batch loss и validation каждые 50 optimizer steps.
`test_generations.json` — все final test inputs и greedy-generated outputs.
`experiments/600_steps/` — сохранённый первый эксперимент и его прежний holdout.
Изменённые seeds или длительность обучения создают новый эксперимент.
