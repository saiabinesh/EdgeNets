#!/bin/sh
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH -A nuig02
#SBATCH -p GpuQ
#SBATCH -o /ichec/work/ngcom027c/logs/train_imagenet_classification_copied_top10_corrected.sh
#SBATCH --mail-user=s.natarajan3@nuigalway.ie
#SBATCH --mail-type=ALL
module load cuda/11.3 gcc conda/2
source activate /ichec/work/nuig02/saiabinesh/pytorch4
cd /ichec/work/nuig02/saiabinesh/EdgeNets
export OMP_NUM_THREADS=1
CUDA_VISIBLE_DEVICES=0,1 python -u train_classification.py --model espnetv2 --s 2.0 --dataset imagenet --data /ichec/work/ngcom027c/Datasets/imgnet_subsets_size_corrected/imgnet_top_10 --scale 0.2 1.0 --scheduler hybrid --lr 0.01 --clr-max 10 --batch-size 144 --epochs 100 --num_classes 10
