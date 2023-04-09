# eecs598-ood-adversarial

To finetune model:
```python
python glue_freelb.py --data_dir glue_data/QNLI/processed --model_type distilbert --model_name_or_path distilbert-base-uncased --task_name qnli --output_dir output/ --do_train --num_train_epochs <epochs> --do_eval
```

finetune.sh contains hyperparameters for finetuning
- set adv_lr and adv_mag to 0 for baseline (no adversarial finetuning)

## Evaluation Results
| Adv. Learning Rate | MNLI       | MNLI-MM    | SNLI       | QNLI       | RTE        |
|--------------------|------------|------------|------------|------------|------------|
| 0 (Baseline)       | 81.54%     | 82.38%     | 75.12%     | **51.37%** | 67.03%     |
| 5e-2               | 82.09%     | 83.06%     | 74.59%     | 51.28%     | 64.49%     |
| 1e-2               | **82.59%** | **83.45%** | **75.80%** | 51.19%     | 66.66%     |
| 5e-3               | 82.04%     | 82.71%     | 75.20%     | 51.15%     | **69.93%** |

## TODOs: