#!/bin/sh
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH -A ngcom020c
#SBATCH -p GpuQ
#SBATCH -o /ichec/work/ngcom020c/Sai/RQ3/logs/train_coco_new_repo_40_pt3_300epochs_1e-2LR.txt
#SBATCH --mail-user=s.natarajan3@nuigalway.ie
#SBATCH --mail-type=BEGIN,END
module load cuda/11.3 gcc conda/2
source activate /ichec/work/ngcom020c/Sai/condaenvs/pytorch3
cd /ichec/work/ngcom020c/Sai/RQ3/Edgenets2/EdgeNets
export OMP_NUM_THREADS=1
CUDA_VISIBLE_DEVICES=0 python -u train_detection.py --model espnetv2 --s 2.0 --dataset coco --data-path ./vision_datasets/coco --lr-type hybrid --lr 0.01 --clr-max 61 --batch-size 32 --epochs 300 --im-size 300


