#!/bin/bash
#PBS -j oe
#PBS -o job_ocd.$PBS_JOBID.log

cd $PBS_O_WORKDIR
mkdir -p logs_ocd
exec > logs_ocd/job_ocd.$PBS_JOBID.log 2>&1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate emo_env

python ocd.py \
  --alg_name "${ALG_NAME}" \
  --problem_name "${PROBLEM_NAME}" \
  --n_obj "${N_OBJ}" 
