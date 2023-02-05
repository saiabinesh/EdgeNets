#!/bin/sh
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH -A ngcom027c
#SBATCH -p GpuQ
#SBATCH -o /ichec/work/ngcom027c/logs/train_imagenet_classification_top_80_to100_uncorrected_es.sh
#SBATCH --mail-user=s.natarajan3@nuigalway.ie
#SBATCH --mail-type=ALL
module load cuda/11.3 gcc conda/2
source activate /ichec/work/nuig02/saiabinesh/pytorch4
export OMP_NUM_THREADS=1
cd /ichec/work/ngcom027c/Sai/general_scripts
python create_temp_dataset_by_copying_size_n_imgnt_new.py /ichec/work/ngcom027c/Datasets/ILSVRC/Data/CLS-LOC 90
cd /ichec/work/nuig02/saiabinesh/EdgeNets
CUDA_VISIBLE_DEVICES=0,1 python -u train_classification.py --model espnetv2 --s 2.0 --dataset imagenet --data /ichec/work/ngcom027c/Datasets/imgnet_subsets/imgnet_top_90 --scale 0.2 1.0 --scheduler hybrid --lr 0.01 --clr-max 2 --batch-size 144 --epochs 50 --num_classes 90 > train_top_90_uncorrected.txt
CUDA_VISIBLE_DEVICES=0,1 python -u test_classification_args.py --model espnetv2 --s 2.0 --dataset imagenet --data /ichec/work/ngcom027c/Datasets/imgnet_subsets/imgnet_top_50 --logs-file train_top_90_uncorrected.txt --num-classes 90 > test_top_90_uncorrected.txt

cd /ichec/work/ngcom027c/Sai/general_scripts
python /ichec/work/ngcom027c/Sai/general_scripts/create_temp_dataset_by_copying_size_n_imgnt_new.py /ichec/work/ngcom027c/Datasets/ILSVRC/Data/CLS-LOC 100
cd /ichec/work/nuig02/saiabinesh/EdgeNets
CUDA_VISIBLE_DEVICES=0,1 python -u train_classification.py --model espnetv2 --s 2.0 --dataset imagenet --data /ichec/work/ngcom027c/Datasets/imgnet_subsets/imgnet_top_100 --scale 0.2 1.0 --scheduler hybrid --lr 0.01 --clr-max 2 --batch-size 144 --epochs 50 --num_classes 100 > train_top_100_uncorrected.txt
CUDA_VISIBLE_DEVICES=0,1 python -u test_classification_args.py --model espnetv2 --s 2.0 --dataset imagenet --data /ichec/work/ngcom027c/Datasets/imgnet_subsets/imgnet_top_100 --logs-file train_top_100_uncorrected.txt --num-classes 100 > test_top_100_uncorrected.txt

