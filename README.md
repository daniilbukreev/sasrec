## Установка

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Работа

Обычный запуск 2 слойной нейросети:
```bash 
python3 -m sasrec.train 
```

Для пошаговой сборки:

```bash
python3 -m sasrec.stack_model 
python3 -m sasrec.train_finetune
```

в `stack_model.py` в качестве аргументов указывать 
1. путь к модели  `--input_model_path`
2. сколько в ней блоков `--shallow_blocks`
3. сколько нужно блоков для новой модели (х2) `--deep_blocks`
 
получим `stacked_b{args.deep_blocks}.pt`

в `train_finetune.py` в качестве аргументов указывать 
1. `--num_blocks` количество блоков в новой модели


Базовые гиперпараметры для ML-20M: `emb_dim=256`, `n_blocks=2`, `n_heads=1`, `dropout=0.1`, `max_len=200`, `lr=1e-3`, `batch=128`, `max_epochs=100`.
