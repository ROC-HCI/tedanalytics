#!/bin/bash
#SBATCH -o out/out.%j.txt -t 5-00:00:00
#SBATCH -c 1
#SBATCH --mem-per-cpu=4gb
#SBATCH -J kamrul
#SBATCH -p standard
#SBATCH -a 1-119
module load openface
python generate_img_vec_final.py $SLURM_ARRAY_TASK_ID
