# eecs598-ood-adversarial

To finetune model:
```python
python glue_freelb.py --data_dir glue_data/QNLI/processed --model_type distilbert --model_name_or_path distilbert-base-uncased --task_name qnli --output_dir output/ --do_train --num_train_epochs <epochs> --do_eval
```

## TODOs:
- Distilbert hardcoded in some places in ``glue_freelb.py``, need to change those if we want to use another model