#!/bin/sh
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH -A ngcom027c
#SBATCH -p GpuQ
#SBATCH -o /ichec/work/ngcom027c/logs/train_imagenet_classification_top50_to_100_uncorrected_es.sh
#SBATCH --mail-user=s.natarajan3@nuigalway.ie
#SBATCH --mail-type=ALL
module load cuda/11.3 gcc conda/2
source activate /ichec/work/nuig02/saiabinesh/pytorch4
export OMP_NUM_THREADS=1
cd /ichec/work/ngcom027c/Sai/general_scripts
python create_temp_dataset_by_copying_size_n_imgnt_new.py /ichec/work/ngcom027c/Datasets/ILSVRC/Data/CLS-LOC 50
cd /ichec/work/nuig02/saiabinesh/EdgeNets
CUDA_VISIBLE_DEVICES=0,1 python -u train_classification.py --model espnetv2 --s 2.0 --dataset imagenet --data /ichec/work/ngcom027c/Datasets/imgnet_subsets/imgnet_top_50 --scale 0.2 1.0 --scheduler hybrid --lr 0.01 --clr-max 2 --batch-size 144 --epochs 50 --num_classes 50 > train_top_50_uncorrected.txt
CUDA_VISIBLE_DEVICES=0,1 python -u test_classification_args.py --model espnetv2 --s 2.0 --dataset imagenet --data /ichec/work/ngcom027c/Datasets/imgnet_subsets/imgnet_top_50 --logs-file /ichec/work/ngcom027c/logs/train_imagenet_classification_top50_to_100_uncorrected_es.sh --num-classes 50 > test_top_50_uncorrected.txt

cd /ichec/work/ngcom027c/Sai/general_scripts
python /ichec/work/ngcom027c/Sai/general_scripts/create_temp_dataset_by_copying_size_n_imgnt_new.py /ichec/work/ngcom027c/Datasets/ILSVRC/Data/CLS-LOC 60
cd /ichec/work/nuig02/saiabinesh/EdgeNets
CUDA_VISIBLE_DEVICES=0,1 python -u train_classification.py --model espnetv2 --s 2.0 --dataset imagenet --data /ichec/work/ngcom027c/Datasets/imgnet_subsets/imgnet_top_60 --scale 0.2 1.0 --scheduler hybrid --lr 0.01 --clr-max 2 --batch-size 144 --epochs 50 --num_classes 60 > train_top_60_uncorrected.txt
CUDA_VISIBLE_DEVICES=0,1 python -u test_classification_args.py --model espnetv2 --s 2.0 --dataset imagenet --data /ichec/work/ngcom027c/Datasets/imgnet_subsets/imgnet_top_60 --logs-file /ichec/work/ngcom027c/logs/train_imagenet_classification_top50_to_100_uncorrected_es.sh --num-classes 60 > test_top_60_uncorrected.txt

cd /ichec/work/ngcom027c/Sai/general_scripts
python /ichec/work/ngcom027c/Sai/general_scripts/create_temp_dataset_by_copying_size_n_imgnt_new.py /ichec/work/ngcom027c/Datasets/ILSVRC/Data/CLS-LOC 70
cd /ichec/work/nuig02/saiabinesh/EdgeNets
CUDA_VISIBLE_DEVICES=0,1 python -u train_classification.py --model espnetv2 --s 2.0 --dataset imagenet --data /ichec/work/ngcom027c/Datasets/imgnet_subsets/imgnet_top_70 --scale 0.2 1.0 --scheduler hybrid --lr 0.01 --clr-max 2 --batch-size 144 --epochs 50 --num_classes 70 > train_top_70_uncorrected.txt
CUDA_VISIBLE_DEVICES=0,1 python -u test_classification_args.py --model espnetv2 --s 2.0 --dataset imagenet --data /ichec/work/ngcom027c/Datasets/imgnet_subsets/imgnet_top_70 --logs-file /ichec/work/ngcom027c/logs/train_imagenet_classification_top50_to_100_uncorrected_es.sh --num-classes 70 > test_top_70_uncorrected.txt


cd /ichec/work/ngcom027c/Sai/general_scripts
python /ichec/work/ngcom027c/Sai/general_scripts/create_temp_dataset_by_copying_size_n_imgnt_new.py /ichec/work/ngcom027c/Datasets/ILSVRC/Data/CLS-LOC 80
cd /ichec/work/nuig02/saiabinesh/EdgeNets
CUDA_VISIBLE_DEVICES=0,1 python -u train_classification.py --model espnetv2 --s 2.0 --dataset imagenet --data /ichec/work/ngcom027c/Datasets/imgnet_subsets/imgnet_top_80 --scale 0.2 1.0 --scheduler hybrid --lr 0.01 --clr-max 2 --batch-size 144 --epochs 50 --num_classes 80 > train_top_80_uncorrected.txt
CUDA_VISIBLE_DEVICES=0,1 python -u test_classification_args.py --model espnetv2 --s 2.0 --dataset imagenet --data /ichec/work/ngcom027c/Datasets/imgnet_subsets/imgnet_top_80 --logs-file /ichec/work/ngcom027c/logs/train_imagenet_classification_top50_to_100_uncorrected_es.sh --num-classes 80 >test_top_80_uncorrected.txt