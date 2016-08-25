#!/bin/bash

#SBATCH --time=0:00:45   # walltime
#SBATCH --ntasks=3   # number of processor cores (i.e. tasks)
#SBATCH --nodes=3   # number of nodes
#SBATCH --mem-per-cpu=1250M   # memory per CPU core
#SBATCH -J "TF.SLURM.MANAGER"   # job name
#SBATCH --gres=gpu:1
#SBATCH --qos=dw87   # S B A T C H  - - q o s = s t a n d b y 
#SBATCH --gid=fslg_pccl

# Compatibility variables for PBS. Delete if not needed.
export PBS_NODEFILE=`/fslapps/fslutils/generate_pbs_nodefile`
export PBS_JOBID=$SLURM_JOB_ID
export PBS_O_WORKDIR="$SLURM_SUBMIT_DIR"
export PBS_QUEUE=batch

# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
source $HOME/fsl_groups/fslg_pccl/configs/group_bashrc

srun python $HOME/fsl_groups/fslg_pccl/projects/modDNN/main.py

