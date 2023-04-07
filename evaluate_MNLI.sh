#!/bin/bash

#SBATCH --job-name=evaluate_MNLI-freelb_CN-E
#SBATCH --account=eecs598w23_class
#SBATCH --mail-user=aakashdp@umich.edu
#SBATCH --mail-type=BEGIN,END

#SBATCH --time=00:10:00

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-gpu=90G
#SBATCH --output=/home/aakashdp/eecs598/eecs598-ood-adversarial/slurm_output/%x-%j.out

# set up job
module load python pytorch cuda/11.6.2
cd /home/aakashdp/eecs598/eecs598-ood-adversarial
source env/bin/activate

TASK_NAME=QNLI  # hack to get the number of labels right without much effort
DATA_DIR=glue_data/MNLI/binary_CN-E

MODEL_TYPE=distilbert                        # Model type
MODEL_NAME_OR_PATH=checkpoints/freelb_trained/checkpoint-best     # Specific huggingface model
OUTPUT_DIR=checkpoints/freelb_trained/checkpoint-best
SEQ_LEN=512                                  # Maximum sequence length

python glue_freelb.py \
  --model_type $MODEL_TYPE \
  --model_name_or_path $MODEL_NAME_OR_PATH \
  --task_name $TASK_NAME \
  --do_eval \
  --do_lower_case \
  --data_dir $DATA_DIR \
  --max_seq_length $SEQ_LEN \
  --output_dir $OUTPUT_DIR

echo Finished evaluation