#!/bin/bash

#OAR -q besteffort
#OAR -l host=1/gpu=1,walltime=3:00:00
#OAR -p kinovis
#OAR -O OAR_%jobid%.out
#OAR -E OAR_%jobid%.err 

# display some information about attributed resources
hostname 
nvidia-smi 
module load conda 
# make use of a python torch environment
conda activate tinyml
# run the training script
python train_single.py
# run the validation script