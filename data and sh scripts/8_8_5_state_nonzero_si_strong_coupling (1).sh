#!/bin/bash
#SBATCH -c 1
#SBATCH -t 10-00:00
#SBATCH -p long
#SBATCH --mem=5000M
#SBATCH -o 8_8_5_state_nonzero_si_strong_coupling__%A_%a.out
#SBATCH -e 8_8_5_state_nonzero_si_strong_coupling__%A_%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lisa_duan@brown.edu
#SBATCH --array=1-3


python3 Desktop/markov/8_8_5_state_nonzero_si_strong_coupling.py ${SLURM_ARRAY_TASK_ID}

