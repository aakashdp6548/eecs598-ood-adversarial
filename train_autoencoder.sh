#!/bin/bash

#SBATCH --job-name=train_autoencoder
#SBATCH --account=eecs598w23_class
#SBATCH --mail-user=aakashdp@umich.edu
#SBATCH --mail-type=BEGIN,END

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=02:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-gpu=8G
#SBATCH --output=/home/aakashdp/eecs598/eecs598-ood-adversarial/slurm_output/%x-%j.out

# set up job
module load python pytorch cuda/11.6.2
cd /home/aakashdp/eecs598/eecs598-ood-adversarial
source env/bin/activate
cd text-autoencoders

DATA_PATH=data/mnli
d=`date "+%Y-%m-%d_%H-%M-%S"`
OUTPUT_DIR="checkpoints/aae-"$d

NUM_EPOCHS=30
BATCH_SIZE=128
SEED=598
MODEL_TYPE=aae

python train.py \
    --train $DATA_PATH/train.txt \
    --valid $DATA_PATH/dev.txt \
    --model_type $MODEL_TYPE \
    --lambda_adv 10 \
    --noise "0.3,0,0,0" \
    --save-dir $OUTPUT_DIR \
    --epochs $NUM_EPOCHS \
    --batch-size $BATCH_SIZE \
    --seed $SEED \
    --vocab-size 50000 \
    --dim_z 128 \
    --dim_emb 512 \
    --dim_h 1024

