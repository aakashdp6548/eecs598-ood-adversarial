#!/bin/bash
# The interpreter used to execute the script

#“#SBATCH” directives that convey submission options:

#SBATCH --job-name=demo
#SBATCH --account=eecs598w23_class
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=00:01:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=10m
#SBATCH --mail-type=BEGIN,END
#SBATCH --output=/home/aakashdp/eecs598/eecs598-ood-adversarial/slurm_output/%x-%j.out

# The application(s) to execute along with its input arguments and options:

/bin/hostname
echo “hello world”
