#!/bin/sh
#SBATCH -p standard
#SBATCH --mem=1gb
#SBATCH -c 1
#SBATCH -a 0
#SBATCH -t 0-1:00:00  
#SBATCH -J sentence_segmenter_echowdh2
#SBATCH -o /scratch/echowdh2/output/segmenter_output%j
#SBATCH -e /scratch/echowdh2/output/segmenter_error%j
#SBATCH --mail-type=all    

module load anaconda
module load htk
module load ffmpeg
python Find_time_segment_in_each_video.py
#python test_environ.py
