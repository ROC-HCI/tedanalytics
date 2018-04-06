#!/bin/sh
#SBATCH -p standard
#SBATCH --mem=4gb
#SBATCH -c 1
#SBATCH -a 0-29
#SBATCH -t 1-:00:00  
#SBATCH -J praat_parsing_whole_echowdh2
#SBATCH -o /scratch/echowdh2/praat_output/whole_output%j
#SBATCH -e /scratch/echowdh2/praat_output/whole_error%j
#SBATCH --mail-type=all    

module load anaconda
module load ffmpeg
python praat_parsing_for_whole_video.py
#python test_environ.py
