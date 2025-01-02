#!/bin/bash
#SBATCH --job-name=llava 	# Job name
#SBATCH --partition=dgx	#Partition name can be test/small/medium/large/gpu #Partition “gpu” should be used only for gpu jobs
#SBATCH --nodes=1 			# Run all processes on a single node
#SBATCH --ntasks=1 			# Run a single task
#SBATCH --cpus-per-task=3 	# Number of CPU cores per task
#SBATCH --gres=gpu:3		# Include gpu for the task (only for GPU jobs)
#SBATCH --mem=120gb 			# Total memory limit
#SBATCH --time=150:00:00 # Time limit hrs:min:sec
#SBATCH --output=output.log # Standard output and error log
date;hostname;pwd
nvidia-smi
eval "$(conda shell.bash hook)"
conda activate patentlmm
sh scripts/v1_5/train_patentlmm_stage1.sh
