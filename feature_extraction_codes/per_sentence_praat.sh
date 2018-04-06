#!/bin/sh
#SBATCH -p standard
#SBATCH --mem=4gb
#SBATCH -c 1
#SBATCH -a 0-29
#SBATCH -t 0-:12:00  
#SBATCH -J praat_parsing_per_sentence_echowdh2
#SBATCH -o /scratch/echowdh2/per_sentence_praat_output/per_sentence_output%j
#SBATCH -e /scratch/echowdh2/per_sentence_praat_output/per_sentence_error%j
#SBATCH --mail-type=all    

module load anaconda
module load ffmpeg
python per_sentence_praat_parser.py

