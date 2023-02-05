#!/bin/sh
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH -A ngcom027c
#SBATCH -p GpuQ
#SBATCH -o /ichec/work/ngcom027c/logs/train_imagenet_classification_top_10_uncorrected_es.sh
#SBATCH --mail-user=s.natarajan3@nuigalway.ie
#SBATCH --mail-type=ALL
module load cuda/11.3 gcc conda/2
source activate /ichec/work/nuig02/saiabinesh/pytorch4
export OMP_NUM_THREADS=1
cd /ichec/work/nuig02/saiabinesh/EdgeNets
CUDA_VISIBLE_DEVICES=0,1 python -u train_classification.py --model espnetv2 --s 2.0 --dataset imagenet --data /ichec/work/ngcom027c/Datasets/imgnet_subsets/imgnet_top_20 --scale 0.2 1.0 --scheduler hybrid --lr 0.01 --clr-max 30 --batch-size 144 --epochs 100 --num_classes 20 > train_top_20_uncorrected_clr_30_100_epochs.txt
CUDA_VISIBLE_DEVICES=0,1 python -u test_classification_args.py --model espnetv2 --s 2.0 --dataset imagenet --data /ichec/work/ngcom027c/Datasets/imgnet_subsets/imgnet_top_20 --logs-file train_top_20_uncorrected_clr_30_100_epochs.txt --num-classes 20 > test_top_20_uncorrected_clr_30_100_epochs.txt

CUDA_VISIBLE_DEVICES=0,1 python -u train_classification.py --model espnetv2 --s 2.0 --dataset imagenet --data /ichec/work/ngcom027c/Datasets/imgnet_subsets/imgnet_top_30 --scale 0.2 1.0 --scheduler hybrid --lr 0.01 --clr-max 30 --batch-size 144 --epochs 100 --num_classes 30 > train_top_30_uncorrected_clr_30_100_epochs.txt
CUDA_VISIBLE_DEVICES=0,1 python -u test_classification_args.py --model espnetv2 --s 2.0 --dataset imagenet --data /ichec/work/ngcom027c/Datasets/imgnet_subsets/imgnet_top_30 --logs-file train_top_30_uncorrected_clr_30_100_epochs.txt --num-classes 30 > test_top_30_uncorrected_clr_30_100_epochs.txt
