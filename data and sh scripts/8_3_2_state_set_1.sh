#!/bin/bash
#SBATCH -c 4
#SBATCH -t 10-00:00
#SBATCH -p long
#SBATCH --mem=5000M
#SBATCH -o 8_3_2_state_set_1_%A_%a_%j.out
#SBATCH -e 8_3_2_state_set_1_%A_%a_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lisa_duan@brown.edu
#SBATCH --array=1-10


python3 Desktop/markov/8_3_2_state_set_1.py ${SLURM_ARRAY_TASK_ID}

