#!/bin/bash

#SBATCH --job-name=finetune_qnli
#SBATCH --account=eecs598w23_class
#SBATCH --mail-user=aakashdp@umich.edu
#SBATCH --mail-type=BEGIN,END

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-gpu=90G
#SBATCH --output=/home/aakashdp/eecs598/eecs598-ood-adversarial/slurm_output/%x-%j.out

# set up job
module load python pytorch cuda/11.6.2
cd /home/aakashdp/eecs598/eecs598-ood-adversarial
source env/bin/activate

TASK_NAME=QNLI
GLUE_DIR=glue_data

MODEL_TYPE=distilbert                        # Model type
MODEL_NAME_OR_PATH=distilbert-base-uncased   # Specific huggingface model
ADV_LR=0 #5e-2                                  # Adversarial step size
ADV_MAG=0 #1e-1                                 # Magnitude of perturbation
ADV_MAX_NORM=0                               # Maximum perturbation
ADV_STEPS=1 #3                                  # Number of adversarial steps
SEQ_LEN=512                                  # Maximum sequence length
LR=1e-5                                      # Learning rate
BATCH_SIZE=8                                 # Batch size
GRAD_ACCU=4                                  # Gradient accumulation steps
#TRAIN_STEPS=33112                            # Number of training steps (parameter updates)
NUM_EPOCHS=3                                 # Number of epochs to train for
WARM_UP_STEPS=1986                           # Learning rate warm-up steps
SEED=42                                      # Random seed
WEIGHT_DECAY=1e-2                            # Weight decay

d=`date "+%Y-%m-%d_%H-%M-%S"`

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
  --output_dir checkpoints/baseline_no_adv \
  --adv-lr $ADV_LR --adv-init-mag $ADV_MAG --adv-max-norm $ADV_MAX_NORM --adv-steps $ADV_STEPS \
  --evaluate_during_training \
  --num_train_epochs $NUM_EPOCHS --warmup_steps $WARM_UP_STEPS --seed $SEED \
  --logging_steps 100 --save_steps 100 \

# runexp TASK_NAME  gpu      model_name      adv_lr  adv_mag  anorm  asteps  lr     bsize  grad_accu  hdp  adp      ts     ws     seed      wd
# runexp QNLI        0       albert-xxlarge-v2   5e-2    1e-1      0    3    1e-5      8       4        0.1   0    33112   1986     42     1e-2


echo Finished finetuning