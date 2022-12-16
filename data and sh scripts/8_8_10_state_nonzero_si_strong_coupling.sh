#!/bin/bash
#SBATCH -c 1
#SBATCH -t 10-00:00
#SBATCH -p long
#SBATCH --mem=5000M
#SBATCH -o 8_8_10_state_nonzero_si_strong_coupling_%j.out
#SBATCH -e 8_8_10_state_nonzero_si_strong_coupling_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lisa_duan@brown.edu


python3 Desktop/markov/8_8_10_state_nonzero_si_strong_coupling.py
