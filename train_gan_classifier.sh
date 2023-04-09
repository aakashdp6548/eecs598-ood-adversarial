#!/bin/bash

#SBATCH --job-name=train_gan_classifier
#SBATCH --account=eecs598w23_class
#SBATCH --mail-user=aakashdp@umich.edu
#SBATCH --mail-type=BEGIN,END

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-gpu=10G
#SBATCH --output=/home/aakashdp/eecs598/eecs598-ood-adversarial/slurm_output/%x-%j.out

# set up job
module load python pytorch cuda/11.6.2
cd /home/aakashdp/eecs598/eecs598-ood-adversarial
source env/bin/activate
cd GAN_text

DATA_PATH=data/classifier
SAVE_PATH=models/mnli

NUM_EPOCHS=50
BATCH_SIZE=32
LR=3e-5
SEED=1111
MODEL_TYPE=emb

python train_baseline.py \
    --data_path $DATA_PATH \
    --save_path $SAVE_PATH \
    --epochs $NUM_EPOCHS \
    --model_type $MODEL_TYPE