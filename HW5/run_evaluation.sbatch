#!/bin/bash
#SBATCH --job-name=eval_jt
#SBATCH --output=eval.out
#SBATCH --error=eval.err
#SBATCH --time=02:00:00
#SBATCH --partition=broadwl
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=14
#SBATCH --mem-per-cpu=4000

source ./test_env/bin/activate
module load python/3.6.1+intel-16.0

python3 evaluate.py

deactivate
