#!/bin/sh
#SBATCH --time=10:30:00
#SBATCH --nodes=1
#SBATCH -A ngcom020c
#SBATCH -p GpuQ
#SBATCH -o /ichec/work/ngcom020c/Sai/RQ3/logs/train_coco_new_repo_pt_OLD_env.txt
#SBATCH --mail-user=s.natarajan3@nuigalway.ie
#SBATCH --mail-type=BEGIN,END
module load cuda/11.2 gcc conda/2
source activate /ichec/work/ngcom020c/Sai/condaenvs/pytorch 
cd /ichec/work/ngcom020c/Sai/RQ3/EdgeNets2/Edgenets
export OMP_NUM_THREADS=1
CUDA_VISIBLE_DEVICES=0 python train_detection.py --model espnetv2 --s 2.0 --dataset coco --data-path ./vision_datasets/coco --lr-type hybrid --lr 0.01 --clr-max 61 --batch-size 64 --epochs 2 --im-size 300
