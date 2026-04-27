#!/bin/bash -l

# Set SCC project
#$ -P ds596

# Request 8 CPUs
#$ -pe omp 8

# Request 4 GPU 
#$ -l gpus=4

# Specify the minimum GPU compute capability. 
#$ -l gpu_c=8.0

# As an example, use the academic-ml module to get Python with machine learning.

module load miniconda

conda activate EpiModX

torchrun --nproc_per_node=3 train_MTL_Moe_parallel.py  -w True --save_model True 
