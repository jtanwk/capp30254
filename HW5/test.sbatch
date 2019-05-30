#!/bin/bash
#SBATCH --job-name=test_jtan
#SBATCH --output=output/test_run.out
#SBATCH --error=output/test_run.err
#SBATCH --time=00:05:00
#SBATCH --partition=broadwl
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=14
#SBATCH --mem-per-cpu=2000

module load python/3.6.1+intel-16.0
python -mvenv test_env
source ./test_env/bin/activate

pip3 install --upgrade pip
pip3 install -r requirements.txt

python3 hw5_pipeline.py

deactivate
