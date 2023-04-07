# eecs598-ood-adversarial

To finetune model:
```python
python glue_freelb.py --data_dir glue_data/QNLI/processed --model_type distilbert --model_name_or_path distilbert-base-uncased --task_name qnli --output_dir output/ --do_train --num_train_epochs <epochs> --do_eval
```

finetune.sh contains hyperparameters for finetuning
- set adv_lr and adv_mag to 0 for baseline (no adversarial finetuning)

## TODOs: