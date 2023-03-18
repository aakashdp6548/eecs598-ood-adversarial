#!/bin/bash

#SBATCH --job-name=evaluate
#SBATCH --account=eecs598w23_class
#SBATCH --mail-user=aakashdp@umich.edu
#SBATCH --mail-type=BEGIN,END

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=00:30:00
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
SEQ_LEN=512                                  # Maximum sequence length

python glue_freelb.py \
  --model_type $MODEL_TYPE \
  --model_name_or_path $MODEL_NAME_OR_PATH \
  --task_name $TASK_NAME \
  --do_eval \
  --do_lower_case \
  --data_dir $GLUE_DIR/$TASK_NAME/processed \
  --max_seq_length $SEQ_LEN \
  --output_dir checkpoints/first_test \

echo Finished evaluation