#!/bin/sh
#SBATCH --job-name=ctp-train
#SBATCH --array=0-3
#SBATCH --output=./output-train-%A_%a.log
#SBATCH --error=./error-train-%A_%a.log

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
##SBATCH --partition=RTX6000Node
#SBATCH --partition=GPUNodes
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=enforce-binding

#SBATCH --mail-type=ALL
#SBATCH --mail-user=jesus.lovon@irit.fr

module purge
module load singularity/3.0.3

_container="/logiciels/containerCollections/CUDA11/pytorch-NGC-20-06-py3.sif"
_python="/users/cost/jlovonme/envs/ctp/bin/python"

configs=( 'WN_par_roberta_large_1e6_seed2' 'WN_par_roberta_large_1e6_wordnet_defs_seed2' 'WN_par_bert_1e6_seed2' 'WN_par_bert_1e6_wordnet_defs_seed2' )

export PYTHONPATH="."
export TOKENIZERS_PARALLELISM=false

cd scripts
srun singularity exec ${_container} ${_python} run_finetuning_hypernym_classification_multidomain.py --experiment-name ${configs[$SLURM_ARRAY_TASK_ID]}
wait

# 2 step - reconciliation
cd ../ctp/inference
srun singularity exec ${_container} ${_python} examine_subtrees.py --prediction-metric-type ancestor --experiment-name ${configs[$SLURM_ARRAY_TASK_ID]}
