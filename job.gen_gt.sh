#!/bin/bash

#SBATCH -J job_name
#SBATCH -A r00068
#SBATCH -o job_logs/filename_%j.txt
#SBATCH -e job_logs/filename_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=13:00:00
#SBATCH --mem=16gb

#Load any modules that your program needs
module load deeplearning/2.9.1

#Run your program
python gen_gt_bev.py