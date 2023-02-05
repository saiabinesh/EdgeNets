#!/bin/sh
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH -A ngcom027c
#SBATCH -p GpuQ
#SBATCH -o /ichec/work/ngcom027c/logs/train_imagenet_top_200_uncorrected_120_epochs.sh
#SBATCH --mail-user=s.natarajan3@nuigalway.ie
#SBATCH --mail-type=ALL
module load cuda/11.3 gcc conda/2
source activate /ichec/work/nuig02/saiabinesh/pytorch4
export OMP_NUM_THREADS=1
cd /ichec/work/nuig02/saiabinesh/EdgeNets

python create_temp_dataset_by_moving_size_n_imgnt_new.py /ichec/work/ngcom027c/Datasets/ILSVRC/Data/CLS-LOC 200

CUDA_VISIBLE_DEVICES=0,1 python -u train_classification.py --model espnetv2 --s 2.0 --dataset imagenet --data /ichec/work/ngcom027c/Datasets/imgnet_subsets/imgnet_top_200 --scale 0.2 1.0 --scheduler hybrid --lr 0.1 --clr-max 61 --batch-size 144 --epochs 300 --num_classes 200 > train_top_200_uncorrected_120_epochs.txt

CUDA_VISIBLE_DEVICES=0,1 python -u test_classification_args.py --model espnetv2 --s 2.0 --dataset imagenet --data /ichec/work/ngcom027c/Datasets/imgnet_subsets/imgnet_top_200 --logs-file train_top_200_uncorrected_120_epochs.txt --num-classes 200 > test_top_200_uncorrected_120_epochs.txt

python move_back_temp_dataset_by_size_n_imgnt_new.py /ichec/work/ngcom027c/Datasets/ILSVRC/Data/CLS-LOC 200
