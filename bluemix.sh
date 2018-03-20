#!/bin/sh
#SBATCH -p standard
#SBATCH --mem=1gb
#SBATCH -c 1
#SBATCH -a 0-75
#SBATCH -t 01:00:00  
#SBATCH -J bmix_mtanveer
#SBATCH -o /scratch/mtanveer/output/bmix_output%j
#SBATCH -e /scratch/mtanveer/output/bmix_error%j

module load anaconda
python -c "import bluemix;bluemix.process_bluehive()"

