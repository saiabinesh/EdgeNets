#!/bin/sh
#SBATCH --time=30:30:00
#SBATCH --nodes=1
#SBATCH -A nuig02
#SBATCH -p GpuQ
#SBATCH -o /ichec/work/ngcom020c/Sai/RQ3/logs/train_handsynth_120_hybrid_500_epochs.txt
#SBATCH --mail-user=s.natarajan3@nuigalway.ie
#SBATCH --mail-type=BEGIN,END
module load cuda/11.2 gcc conda/2
source activate /ichec/work/ngcom020c/Sai/condaenvs/pytorch
cd /ichec/work/ngcom020c/Sai/RQ3/EdgeNets
export OMP_NUM_THREADS=1
CUDA_VISIBLE_DEVICES=0 python train_detection.py --model espnetv2 --s 2.0 --dataset handsynth --data-path /ichec/work/ngcom020c/Sai/RQ3/Datasets/detection_V2 --lr-type hybrid --lr 0.005 --clr-max 150 --batch-size 32 --epochs 600 --im-size 512  
