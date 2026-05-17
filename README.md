## Установка

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Работа

Обычный запуск 2 слойной нейросети:
```bash 
python3 -m sasrec.train \ ```

Для пошаговой сборки:

```bash
python3 -m sasrec.stack_blocks \
python3 -m sasrec.train_finetune \ ```

в соответствии с input_rules.txt

Базовые гиперпараметры для ML-20M: `emb_dim=256`, `n_blocks=2`, `n_heads=1`, `dropout=0.1`, `max_len=200`, `lr=1e-3`, `batch=128`, `max_epochs=100`.
