#!/bin/bash

#OAR -q production
#OAR -l host=1/gpu=1,walltime=3:00:00
#OAR -p gpu-16GB AND gpu_compute_capability_major>=5
#OAR -O OAR_%jobid%.out
#OAR -E OAR_%jobid%.err 

# display some information about attributed resources
hostname 
nvidia-smi 
 
# make use of a python torch environment
conda activate tinyml

# run the training script
python train_single.py
# run the validation script