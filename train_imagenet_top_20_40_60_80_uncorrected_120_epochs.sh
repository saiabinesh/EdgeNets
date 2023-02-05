#!/bin/sh
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH -A ngcom027c
#SBATCH -p GpuQ
#SBATCH -o /ichec/work/ngcom027c/logs/train_imagenet_top_20_40_60_80_uncorrected_120_epochs.sh
#SBATCH --mail-user=s.natarajan3@nuigalway.ie
#SBATCH --mail-type=ALL
module load cuda/11.3 gcc conda/2
source activate /ichec/work/nuig02/saiabinesh/pytorch4
export OMP_NUM_THREADS=1
cd /ichec/work/nuig02/saiabinesh/EdgeNets

CUDA_VISIBLE_DEVICES=0,1 python -u train_classification.py --model espnetv2 --s 2.0 --dataset imagenet --data /ichec/work/ngcom027c/Datasets/imgnet_subsets/imgnet_top_20 --scale 0.2 1.0 --scheduler hybrid --lr 0.1 --clr-max 61 --batch-size 144 --epochs 120 --num_classes 20 > train_top_20_uncorrected_120_epochs.txt

CUDA_VISIBLE_DEVICES=0,1 python -u test_classification_args.py --model espnetv2 --s 2.0 --dataset imagenet --data /ichec/work/ngcom027c/Datasets/imgnet_subsets/imgnet_top_20 --logs-file train_top_20_uncorrected_120_epochs.txt --num-classes 20 > test_top_20_uncorrected_120_epochs.txt

CUDA_VISIBLE_DEVICES=0,1 python -u train_classification.py --model espnetv2 --s 2.0 --dataset imagenet --data /ichec/work/ngcom027c/Datasets/imgnet_subsets/imgnet_top_40 --scale 0.2 1.0 --scheduler hybrid --lr 0.1 --clr-max 61 --batch-size 144 --epochs 120 --num_classes 40 > train_top_40_uncorrected_120_epochs.txt

CUDA_VISIBLE_DEVICES=0,1 python -u test_classification_args.py --model espnetv2 --s 2.0 --dataset imagenet --data /ichec/work/ngcom027c/Datasets/imgnet_subsets/imgnet_top_20 --logs-file train_top_40_uncorrected_120_epochs.txt --num-classes 40 > test_top_40_uncorrected_120_epochs.txt

CUDA_VISIBLE_DEVICES=0,1 python -u train_classification.py --model espnetv2 --s 2.0 --dataset imagenet --data /ichec/work/ngcom027c/Datasets/imgnet_subsets/imgnet_top_60 --scale 0.2 1.0 --scheduler hybrid --lr 0.1 --clr-max 61 --batch-size 144 --epochs 120 --num_classes 60 > train_top_60_uncorrected_120_epochs.txt

CUDA_VISIBLE_DEVICES=0,1 python -u test_classification_args.py --model espnetv2 --s 2.0 --dataset imagenet --data /ichec/work/ngcom027c/Datasets/imgnet_subsets/imgnet_top_20 --logs-file train_top_60_uncorrected_120_epochs.txt --num-classes 60 > test_top_60_uncorrected_120_epochs.

CUDA_VISIBLE_DEVICES=0,1 python -u train_classification.py --model espnetv2 --s 2.0 --dataset imagenet --data /ichec/work/ngcom027c/Datasets/imgnet_subsets/imgnet_top_80 --scale 0.2 1.0 --scheduler hybrid --lr 0.1 --clr-max 61 --batch-size 144 --epochs 120 --num_classes 80 > train_top_80_uncorrected_120_epochs.txt

CUDA_VISIBLE_DEVICES=0,1 python -u test_classification_args.py --model espnetv2 --s 2.0 --dataset imagenet --data /ichec/work/ngcom027c/Datasets/imgnet_subsets/imgnet_top_20 --logs-file train_top_80_uncorrected_120_epochs.txt --num-classes 80 > test_top_80_uncorrected_120_epochs.txt

