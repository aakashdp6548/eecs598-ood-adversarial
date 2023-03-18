#!/bin/bash

#SBATCH --job-name=finetune_qnli
#SBATCH --account=eecs598w23_class
#SBATCH --mail-user=aakashdp@umich.edu
#SBATCH --mail-type=BEGIN,END

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=10m
#SBATCH --cpus-per-gpu=20
#SBATCH --mem-per-gpu=90G
#SBATCH --output=/home/aakashdp/eecs598/eecs598_ood_adversarial/slurm_output/%x-%j.out

# set up job
module load python pytorch cuda/11.6.2
pushd /home/aakashdp/eecs598/eecs598_ood_adversarial
source env/bin/activate

export TASK_NAME=QNLI
export GLUE_DIR=glue_data
export JOB=%x-%j

export MODEL_TYPE=distilbert                        # Model type
export MODEL_NAME_OR_PATH=distilbert-base-cased     # Specific huggingface model
export ADV_LR=5e-2                                  # Adversarial step size
export ADV_MAG=1e-1                                 # Magnitude of perturbation
export ADV_MAX_NORM=0                               # Maximum perturbation
export ADV_STEPS=3                                  # Number of adversarial steps
export SEQ_LEN=512                                  # Maximum sequence length
export LR=1e-5                                      # Learning rate
export BATCH_SIZE=8                                 # Batch size
export GRAD_ACCU=4                                  # Gradient accumulation steps
export TRAIN_STEPS=33112                            # Number of training steps (parameter updates)
export WARM_UP_STEPS=1986                           # Learning rate warm-up steps
export SEED=42                                      # Random seed
export WEIGHT_DECAY=1e-2                            # Weight decay

python glue_freelb.py \
  --model_type $MODEL_TYPE \
  --model_name_or_path $MODEL_NAME_OR_PATH \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir $GLUE_DIR/$TASK_NAME/processed \
  --max_seq_length $SEQ_LEN \
  --per_gpu_train_batch_size $BATCH_SIZE --gradient_accumulation_steps $GRAD_ACCU \
  --learning_rate $LR --weight_decay $WEIGHT_DECAY \
  --output_dir checkpoints/ \
  --adv-lr $ADV_LR --adv-init-mag $ADV_MAG --adv-max-norm $ADV_MAX_NORM --adv-steps $ADV_STEPS \
  --evaluate_during_training \
  --max_steps $TRAIN_STEPS --warmup_steps $WARM_UP_STEPS --seed $SEED \
  --logging_steps 100 --save_steps 100 \

# runexp TASK_NAME  gpu      model_name      adv_lr  adv_mag  anorm  asteps  lr     bsize  grad_accu  hdp  adp      ts     ws     seed      wd
# runexp QNLI        0       albert-xxlarge-v2   5e-2    1e-1      0    3    1e-5      8       4        0.1   0    33112   1986     42     1e-2


