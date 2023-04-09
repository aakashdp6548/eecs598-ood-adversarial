#!/bin/bash

#SBATCH --job-name=evaluate
#SBATCH --account=eecs598w23_class
#SBATCH --mail-user=aakashdp@umich.edu
#SBATCH --mail-type=BEGIN,END

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-gpu=5G
#SBATCH --output=/home/aakashdp/eecs598/eecs598-ood-adversarial/slurm_output/%x-%j.out

# set up job
module load python pytorch cuda/11.6.2
cd /home/aakashdp/eecs598/eecs598-ood-adversarial
source env/bin/activate

TASK_NAME=QNLI

MODEL_DIR=checkpoints/mnli_adv_distilbert-2023-04-07_13-13-13
MODEL_CHECKPOINT=$MODEL_DIR/checkpoint-last/
DATA_DIR=glue_data/QNLI/processed

MODEL_TYPE=distilbert                        # Model type
SEQ_LEN=512                                  # Maximum sequence length

python glue_freelb.py \
  --model_type $MODEL_TYPE \
  --model_name_or_path $MODEL_CHECKPOINT \
  --task_name $TASK_NAME \
  --do_eval \
  --do_lower_case \
  --data_dir $DATA_DIR \
  --max_seq_length $SEQ_LEN \
  --output_dir $MODEL_CHECKPOINT

echo Finished evaluation