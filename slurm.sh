#!/bin/bash

#SBATCH --time=0:00:45   # walltime
#SBATCH --ntasks=4   # number of processor cores (i.e. tasks)
#SBATCH --nodes=2   # number of nodes
#SBATCH --mem-per-cpu=125M   # memory per CPU core
#SBATCH -J "TF.slurmmanager.test"   # job name
#SBATCH --gres=gpu:1
#SBATCH --qos=dw87   # S B A T C H  - - q o s = s t a n d b y 
#SBATCH --gid=fslg_pccl

# Compatibility variables for PBS. Delete if not needed.
export PBS_NODEFILE=`/fslapps/fslutils/generate_pbs_nodefile`
export PBS_JOBID=$SLURM_JOB_ID
export PBS_O_WORKDIR="$SLURM_SUBMIT_DIR"
export PBS_QUEUE=batch

source $HOME/fsl_groups/fslg_pccl/configs/group_bashrc
srun python $HOME/fsl_groups/fslg_pccl/projects/modDNN/test_nodelistparser.py

# To use: sbatch slurm.sh